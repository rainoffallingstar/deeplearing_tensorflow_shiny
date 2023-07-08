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

# define functions 

image_flow <- function(x,y,z){
  image_dataset_from_directory(directory = paste0(getwd(), "/", input$filedir, x), 
                               batch_size = y,
                               image_size = z)
}

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
                        value = 0),
          pickerInput("model",
                      "Select a model:",
                      selected = "mobilenet",
                      choices = c("plainCNN", "mobilenet", "resnet",
                                  "effientNet", "dansenet",
                                  "mobilenetV3","xception",
                                  "nasnet")
          ),
          materialSwitch(
            inputId = "Id078",
            label = "Only test mod",
            value = TRUE, 
            status = "danger"
          )
          
          
          
        ),

        # Show a plot of the generated distribution
        mainPanel(
          tabsetPanel(
            tabPanel(
              "训练曲线图", plotOutput('plots'),
              textOutput("status")
            ),
            
            #tabPanel(
             # "训练细节", verbatimTextOutput("history")
            #),
            tabPanel(
              "测试结果", plotOutput("predict"),
              textOutput("auc"),
              textOutput("acc")
              
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
  
  training_image_flow <- reactive({
    image_flow( "/train/",
                10,
                c(224, 224))
  })
  
  validation_image_flow <- reactive({
    image_flow( "/validation/",
                10,
                c(224, 224))
  })
  
  models <- reactive({
    nums <- length(classnum())
    # define main models
    if (input$model == "mobilenet") {
      mob <- application_mobilenet(include_top = FALSE, pooling = "avg")
    } else if (input$model == "effientNet"){
      mob <- application_efficientnet_b0(include_top = FALSE, pooling = "avg")
    } else if (input$model == "resnet") {
      mob <- application_resnet50_v2(include_top = FALSE, pooling = "avg")
    } else if (input$model == "dansenet") {
      mob <- application_densenet(include_top = FALSE, pooling = "avg")
    } else if (input$model == "mobilenetV3"){
      mob <- application_mobilenet_v3_small(include_top = FALSE, pooling = "avg")
    } else if (input$model == "xception" ){
      mob <- application_xception(include_top = FALSE, pooling = "avg")
    } else if (input$model == "nasnet"){
      mob <- application_nasnet(include_top = FALSE, pooling = "avg")
    }else {
      model <- keras_model_sequential() %>% 
        layer_dense(units = 256, activation = "relu") 
    }
     
    unfreeze_weights(mob)
    inputs <-layer_input(shape = c(224, 224, 3) )
    data_augmentation <- keras_model_sequentail() %>% 
      layer_random_flip("horizontal") %>% 
      layer_random_rotation(0.2) %>% 
      layer_random_zoom(0.2) %>% 
      layer_rescaling(1/255)
    outputs <- inputs %>% 
      data_augmentation() %>% 
      mob() %>%
      layer_dense(256, activation = "relu") %>% 
      layer_flatten() %>%
      layer_dropout(rate = input$dropout_factor) %>% 
      layer_dense(units = as.numeric(nums), activation = "softmax")
      
    model <- keras_model(inputs,
                         outputs)
     
  })
  
  output$plots <- renderPlot({
    if (input$Id078 == TRUE) {
      img <- jpeg::readJPEG(paste0(getwd(),"/image/logo.jpg"))
      history <- as.raster(img)
    } else {
      epoches = input$epoch
    
    training_image_flow <- training_image_flow()
    validation_image_flow <- validation_image_flow()
    model <- models()
    withProgress(message = 'Calculation in progress',
                 detail = 'This may take a while...', value = 1, {
    model %>% 
      compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
    # Include the epoch in the file name
    checkpoint_path <- paste0(getwd(),"/model/",input$model, 
                              "/best_weight.keras")
    checkpoint_dir <- fs::path_dir(checkpoint_path)
    # Create a callback that saves the model's weights every 2 epochs
    batch_size = 10
    cp_callback <- callback_model_checkpoint(
      filepath = checkpoint_path,
      save_best_only = TRUE,
      mointor = "val_loss"
    )
    history <- model %>% fit_generator(
        training_image_flow, 
        epochs = epoches, 
        steps_per_epoch = training_image_flow$n/training_image_flow$batch_size,
        validation_data = validation_image_flow,
        validation_steps = validation_image_flow$n/validation_image_flow$batch_size,
        callbacks = list(cp_callback)
      )
                 })
    
    }
    plot(history)
    
  })
  output$status <- renderText({
    if (input$Id078 == TRUE) {
      print("Status: the test mod is on")
    } else {
      print("Status: the training mod is on")
    }
  })
  
  test_model <- reactive({
    load_model_tf(paste0(getwd(),"/model/",input$model, 
                         "/best_weight.keras"))
  })
  
  predict_result <- reactive({
    
    test_model <- test_model()
    
    cla <- classnum()
    df <- data.frame()
    for (i in 1:length(cla)) {
      test_flow <-  image_flow(paste0("/test_",
                            cla[i],"/"), 
                            10,
                            c(224, 224)
      )
      d <- model %>% 
        predict(test_flow, steps = test_flow$n/test_flow$batch_size) 
      if (i == 1){
        result <- data.frame(classes = rep(cla[i],length(d[,i])), prob = d[,i]) 
      }else{
        result <- data.frame(classes = rep(cla[i],length(d[,i])), prob = 1-d[,i]) 
      } 
      df <- rbind(df,result) 
    }
    attach(df)
    df %>% within(
      classes <- factor(classes, levels = cla )
      
    )
    rocobj <- roc(
      df[,1], 
      df[,2],
      smooth = F
    )
})

  
  output$predict <- renderPlot({
    
    withProgress(message = 'Calculation in progress',
                 detail = 'This may take a while...', value = 1, {
                   rocobj <- predict_result()# 计算临界点/阈值
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
                   })
    
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
                  alpha=0.5)#+
      #geom_point(aes(x = cutOffPoint[[2]],y = cutOffPoint[[3]]))+ # 绘制临界点/阈值
      #geom_text(aes(x = cutOffPoint[[2]],y = cutOffPoint[[3]],label=cutOffPointText),vjust=-1) # 添加临界点/阈值文字标签
    
 })
  
  output$auc <- renderText ({
    rocobj <- predict_result()
    auc<-auc(rocobj)[1]
    print(paste("the auc of model:",input$model,"is",auc))
  })
  
  output$acc <- renderText({
    test_model <- test_model()
    test_flow <-  image_flow("/test/", 
                             10,
                             c(224, 224))
    result <- evaluate(test_model,test_flow)
    cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))
  })
  
 
 
}

# Run the application 
shinyApp(ui = ui, server = server)
