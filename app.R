#####
# 说明
# 文件构架：
#- app.R
#|- data (默认调用本地文件夹名，内应含已处理好的train、validatation文件夹)
#|-- train （按文件夹名分类放置）
#|-- validation
#|-- test_{分类名} （默认调用测试文件夹序列，内置默认全分类的空文件夹，及测试对象的文件）
#|- model （默认保存和调用的训练模型文件，以模型名称命名文件夹）
# workflow
# 1. 随机划分测试集患者序列
# 2. 按标签汇总文件，并进行训练集和测试集的增强
# 3. 设置训练过程，epoch设置为30
# 4. 记录结果

library(keras)
library(tfhub)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)
library(pins)
library(fs)
library(shinyWidgets)
library(shiny)
library(pROC)
library(ggplot2)

ui <- fluidPage(

    # Application title
    titlePanel("The Great Deep-Learning Panel"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          textInput("filedir",
                    "input your task path",
                    "data"),
          sliderInput("dropout_factor", 
                      "Set a probability for dropout layer",
                      min = 0,
                      max = 1,
                      value = 0.2),
            sliderInput("epoch",
                        "Number of epoch",
                        min = 0,
                        max = 100,
                        value = 30),
          pickerInput("model",
                      "Select a model:",
                      selected = "plainCNN",
                      choices = c("plainCNN", "mobilenet", "resnet",
                                  "effientNet", "dansenet")
          )
          
        ),

        # Show a plot of the generated distribution
        mainPanel(
          tabsetPanel(
            tabPanel(
              "训练曲线图", plotOutput('plots')
            ),
            
            #tabPanel(
             # "训练细节", verbatimTextOutput("history")
            #),
            tabPanel(
              "测试结果", plotOutput("predict"),
              verbatimTextOutput("aucvalue")
              
            )
            #tabPanel(
              #"ROC图", plotOutput("rocplots")
            #)
            
            
          )
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

  classnum <- reactive({
    classes <- list.dirs(paste0(getwd(), "/", input$filedir, "/train/"),
                         full.names = FALSE, recursive = FALSE)
    
  })
  
  
  training_image_gen <- reactive({
    image_data_generator(
      rotation_range = 20,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      horizontal_flip = TRUE,
      preprocessing_function = imagenet_preprocess_input
    )
  })
  
  validation_image_gen <- reactive({
    image_data_generator(
      preprocessing_function = imagenet_preprocess_input
    )
  })
  
  training_image_flow <- reactive({
    
    flow_images_from_directory(
      directory = paste0(getwd(), "/", input$filedir, "/train/"), 
      generator = training_image_gen(), 
      class_mode = "categorical",
      batch_size = 5,
      target_size = c(224, 224), 
    )
  })
  
  validation_image_flow <- reactive({
    flow_images_from_directory(
      directory = paste0(getwd(), "/", input$filedir, "/validation/"), 
      generator = validation_image_gen(), 
      class_mode = "categorical",
      batch_size = 5,
      target_size = c(224, 224), 
      shuffle = FALSE
    )
  })
  
  models <- reactive({
    nums <- length(classnum())
    if (input$model == "mobilenet") {
      mob <- application_mobilenet(include_top = FALSE, pooling = "avg")
      unfreeze_weights(mob)
      
      model <- keras_model_sequential() %>% 
        mob() %>% 
        layer_dense(256, activation = "relu") %>% 
        layer_dropout(rate = input$dropout_factor) %>% 
        layer_dense(units = as.numeric(nums), activation = "softmax")
    } else if (input$model == "effientNet"){
      feature_extractor_url <- "https://hub.tensorflow.google.cn/google/imagenet/efficientnet_v2_imagenet21k_b0/classification/2"
      feature_extractor_layer <- layer_hub(handle = feature_extractor_url, 
                                           input_shape = c(224, 224))
      unfreeze_weights(feature_extractor_layer)
      model <- keras_model_sequential(list(
        feature_extractor_layer,
        layer_dense(256, activation = "relu"),
        layer_dropout(rate = input$dropout_factor),
        layer_dense(units = as.numeric(nums), activation = "softmax")
      ))
    } else if (input$model == "resnet") {
      feature_extractor_url <- "https://hub.tensorflow.google.cn/google/bit/m-r50x1/ilsvrc2012_classification/1"
      feature_extractor_layer <- layer_hub(handle = feature_extractor_url, 
                                           input_shape = c(224, 224))
      unfreeze_weights(feature_extractor_layer)
      model <- keras_model_sequential(list(
        feature_extractor_layer,
        layer_dense(256, activation = "relu"),
        layer_dropout(rate = input$dropout_factor),
        layer_dense(units = as.numeric(nums), activation = "softmax")
      ))
    } else if (input$model == "dansenet") {
      input_img <- layer_input(shape = c(224, 224, 3))
      mod <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = input$dropout_factor)
      unfreeze_weights(mod)
      model <- keras_model_sequential(list(
        mod,
        layer_dense(units = as.numeric(nums), activation = "softmax")
      ))
    } 
      model <- keras_model_sequential() %>% 
        layer_flatten() %>% 
        layer_dense(units = 256, activation = "relu") %>% 
        layer_dense(units = 128, activation = "relu") %>% 
        layer_dropout(rate = input$dropout_factor) %>% 
        layer_dense(units = as.numeric(nums), activation = "softmax")
    
  })
  
  output$plots <- renderPlot({
    training_image_flow <- training_image_flow()
    validation_image_flow <- validation_image_flow()
    model <- models()
    withProgress(message = 'Calculation in progress',
                 detail = 'This may take a while...', value = 0, {
    model %>% 
      compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
    # Include the epoch in the file name
    checkpoint_path <- paste0(getwd(),"/model/",input$model, 
                              "/cp-list{epoch:04d}.ckpt")
    checkpoint_dir <- fs::path_dir(checkpoint_path)
    # Create a callback that saves the model's weights every 2 epochs
    batch_size = 10
    cp_callback <- callback_model_checkpoint(
      filepath = checkpoint_path,
      verbose = 1,
      save_weights_only = TRUE,
      save_freq = 2*batch_size
    )
    history <- model %>% fit_generator(
        generator = training_image_flow, 
        epochs = input$epoch, 
        steps_per_epoch = training_image_flow$n/training_image_flow$batch_size,
        validation_data = validation_image_flow,
        validation_steps = validation_image_flow$n/validation_image_flow$batch_size,
        callbacks = list(cp_callback)
      )
                 })
    plot(history)
    
  })
  
  predict_result <- reactive({
    checkpoint_path <- paste0(getwd(),"/model/",input$model, 
                              "/cp-list{epoch:04d}.ckpt")
    checkpoint_dir <- fs::path_dir(checkpoint_path)
    latest <- tf$train$latest_checkpoint(checkpoint_dir)
    model <- models()
    load_model_weights_tf(model,latest)
    
    cla <- classnum()
    df <- data.frame()
    for (i in 1:length(cla)) {
      test_flow <-  flow_images_from_directory(
        generator = image_data_generator(),
        directory =  paste0(getwd(), "/", input$filedir, "/test_",
                            cla[i],"/"), 
        target_size = c(224, 224),
        class_mode = "categorical",
        shuffle = FALSE
      )
      d <- model %>% 
        predict(test_flow, steps = test_flow$n/test_flow$batch_size) 
      if (i == 1){
        result <- data.frame(rep(cla[i],length(d[,i])), d[,i]) 
      }else{
        result <- data.frame(rep(cla[i],length(d[,i])), 1-d[,i])
      } 
      df <- rbind(df,result)
      rocobj <- roc(
        df[,1], 
        df[,2],
        smooth = F
      )
    }
  })

  
  output$predict <- renderPlot({
    rocobj <- predict_result()
    # 计算临界点/阈值
    cutOffPoint <- coords(rocobj, "best")
    cutOffPointText <- paste0(round(cutOffPoint[1],3),"(",round(cutOffPoint[2],3),",",round(cutOffPoint[3],3),")")
    
    # 计算AUC值
    auc<-auc(rocobj)[1]
    # AUC的置信区间
    auc_low<-ci(rocobj,of="auc")[1]
    auc_high<-ci(rocobj,of="auc")[3]
    
    # 计算置信区间
    ciobj <- ci.se(rocobj,specificities=seq(0, 1, 0.01))
    data_ci<-ciobj[1:101,1:3]
    data_ci<-as.data.frame(data_ci)
    x=as.numeric(rownames(data_ci))
    data_ci<-data.frame(x,data_ci)
    
    # 绘图
    ggroc(rocobj,
          color="red",
          size=1,
          legacy.axes = F # FALSE时 横坐标为1-0 specificity；TRUE时 横坐标为0-1 1-specificity
    )+
      theme_classic()+
      geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1),        # 绘制对角线
                   colour='grey', 
                   linetype = 'dotdash'
      ) +
      geom_ribbon(data = data_ci,                                # 绘制置信区间
                  aes(x=x,ymin=X2.5.,ymax=X97.5.), 
                  fill = 'lightblue',
                  alpha=0.5)+
      geom_point(aes(x = cutOffPoint[[2]],y = cutOffPoint[[3]]))+ # 绘制临界点/阈值
      geom_text(aes(x = cutOffPoint[[2]],y = cutOffPoint[[3]],label=cutOffPointText),vjust=-1) # 添加临界点/阈值文字标签
    
  })
  
  output$aucvlue <- renderPrint({
    rocobj <- predict_result()
    auc<-auc(rocobj)[1]
  }
    
  )
 
 
}

# Run the application 
shinyApp(ui = ui, server = server)
