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
# load weights from local files
applications_model_local <- function(filepath, custom_objects = NULL, compile = FALSE,include_top = FALSE) {
  message("info::this function is driven from the orginal load_model and load_model_tf functions in the keras for R package")
  # a basic function to load models for .keras format and try to remove the classfier
  load_model <- function(filepath, custom_objects = NULL, compile = TRUE, include_top= TRUE) {
    # prepare custom objects
    custom_objects <- objects_with_py_function_names(custom_objects)
    
    # build args dynamically so we can only pass `compile` if it's supported
    # (compile requires keras 2.0.4 / tensorflow 1.3)
    args <- list(
      filepath = normalize_path(filepath),
      custom_objects = custom_objects
    )
    if (keras_version() >= "2.0.4"){
      args$compile <- compile
      args$include_top <- include_top
    }
    do.call(keras$models$load_model, args)
  }
  if (tensorflow::tf_version() < "2.0.0")
    stop("TensorFlow version >= 2.0.0 is requires to load models in the SavedModel format.",
         call. = FALSE)
  
  mob <- load_model(filepath, custom_objects, compile,include_top)
  return(mob)
}

image_flow <- function(a,x,y,z){
  flow_images_from_directory(
    directory = paste0(getwd(), "/", a,x), 
    generator =  image_data_generator(
      preprocessing_function = imagenet_preprocess_input
    ), 
    class_mode = "categorical",
    batch_size = y,
    target_size = z, 
    shuffle = TRUE
  )
}

tf_read_image <- function(path, format = "image", resize = NULL, ...) {
    img <- path %>%
      tf$io$read_file() %>%
      tf$io[[paste0("decode_", format)]](...)
    if (!is.null(resize))
      img <- img %>%
        tf$image$resize(as.integer(resize))
    img
  }
plot_activations <- function(x, ...) {
  x <- as.array(x)
  if(sum(x) == 0)
    return(plot(as.raster("gray")))
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(x), asp = 1, axes = FALSE, useRaster = TRUE,
        col = terrain.colors(256), ...)
}

display_image_tensor <- function(x, ..., max = 255,
                                 plot_margins = c(0, 0, 0, 0)) {
  if(!is.null(plot_margins))
    par(mar = plot_margins)
  x %>%
    as.array() %>%
    drop() %>%
    as.raster(max = max) %>%
    plot(..., interpolate = FALSE)
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
                  value = 40),
      pickerInput("model",
                  "Select a model:",
                  selected = "mobilenet",
                  choices = c("plainCNN", "mobilenet", "resnet",
                              "effientNet", "dansenet",
                              "mobilenetV3","xception",
                              "nasnet","VisionTransformer","localweight")
      ),
      textInput("localweight",
                "input your localweight for further transfer learning",
                "NULL"),
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
        
        tabPanel(
          "模型构架", verbatimTextOutput("detail_model"),
          verbatimTextOutput("detail_freeze_model")
        ),
        tabPanel(
          "测试结果", plotOutput("predict"),
          textOutput("auc"),
          textOutput("acc")
          
        )
        ,
        #tabPanel(
        #"数据增强示例", plotOutput("dataaug")
        #),
        tabPanel(
          "Grad-Cam(under working)",plotOutput("gradcam")
        )
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
    image_flow( input$filedir,"/train/",
                10,
                c(224, 224))
  })
  
  validation_image_flow <- reactive({
    image_flow( input$filedir,"/validation/",
                10,
                c(224, 224))
  })
  
  mod <- reactive({
    if (input$model == "mobilenet") {
      mob <- application_mobilenet(weights = "imagenet",
                                   include_top = FALSE,
                                   input_shape = c(224, 224,3))
    } else if (input$model == "effientNet"){
      mob <- application_efficientnet_b0(weights = "imagenet",
                                         include_top = FALSE,  
                                         input_shape = c(224, 224,3))
    } else if (input$model == "resnet") {
      mob <- application_resnet50_v2(weights = "imagenet",
                                     include_top = FALSE, 
                                     input_shape = c(224, 224,3))
    } else if (input$model == "dansenet") {
      mob <- application_densenet(weights = "imagenet",
                                  include_top = FALSE, 
                                  input_shape = c(224, 224,3))
    } else if (input$model == "mobilenetV3"){
      mob <- application_mobilenet_v3_small(weights = "imagenet",
                                            include_top = FALSE, 
                                            input_shape = c(224, 224,3))
    } else if (input$model == "xception" ){
      mob <- application_xception(weights = "imagenet",
                                  include_top = FALSE, 
                                  input_shape = c(224, 224,3))
    } else if (input$model == "nasnet"){
      mob <- application_nasnetmobile(weights = "imagenet",
                                      include_top = FALSE, 
                                      input_shape = c(224, 224,3))
    } else if (input$model == "VisionTransformer"){
      feature_extractor_url <- "https://hub.tensorflow.google.cn/sayakpaul/vit_s16_classification/1"
      mob <- layer_hub(handle = feature_extractor_url, 
                       input_shape = c(224, 224,3))
    } else if (input$model == "localweight"){
      mob <- applications_model_local(input$localweight)
    }
    else {
      model <- keras_model_sequential() %>% 
        layer_dense(units = 256, activation = "relu") 
    }
  })
  
  data_augmentations <- reactive({
    data_augmentations <- keras_model_sequential() %>% 
      layer_random_flip() %>% 
      layer_random_rotation(0.2) %>% 
      layer_random_zoom(0.2) 
  })
  
  models <- reactive({
    nums <- length(classnum())
    # define main models
    mob <- mod()
    unfreeze_weights(mob)
    if (input$model == "localweight"){
      model <- keras_model_sequential() %>% 
        mob %>% 
        layer_dense(units = as.numeric(nums), activation = "softmax",name = "predictions")
    }else {
      inputs <-layer_input(shape = c(224, 224,3) )
      data_augmentation <- data_augmentations()
      outputs <- inputs %>% 
        data_augmentation() %>% 
        layer_rescaling(1/255) %>% 
        mob %>%
        layer_dropout(rate = input$dropout_factor,name = "dropout") %>% 
        #layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu",name = "last_conv_layer") %>%
        layer_average_pooling_2d(pool_size = 2,name = "avg_pool") %>%
        layer_flatten() %>%
        layer_dense(units = as.numeric(nums), activation = "softmax",name = "predictions")
      model <- keras_model(inputs,
                           outputs)
    }
    return(model)
    
  })
  
  grad_cam_process <- reactive({
    img_path <- paste0(getwd(),"/image/example.jpg")
    img_tensor <- tf_read_image(img_path, resize = c(224, 224))
    preprocessed_img <- img_tensor[tf$newaxis, , , ] %>%
      imagenet_preprocess_input()
    model <- models()
    last_conv_layer_name <- "last_conv_layer"
    classifier_layer_names <- c("avg_pool", "predictions")
    last_conv_layer <- model %>% get_layer(last_conv_layer_name)
    last_conv_layer_model <- keras_model(model$inputs,
                                         last_conv_layer$output)
    classifier_input <- layer_input(batch_shape = last_conv_layer$output$shape)
    x <- classifier_input
    for (layer_name in classifier_layer_names)
      x <- get_layer(model, layer_name)(x)
    classifier_model <- keras_model(classifier_input, x)
    with (tf$GradientTape() %as% tape, {
     last_conv_layer_output <- last_conv_layer_model(preprocessed_img)
     tape$watch(last_conv_layer_output)
      preds <- classifier_model(last_conv_layer_output)
      top_pred_index <- tf$argmax(preds[1, ])
      top_class_channel <- preds[, top_pred_index, style = "python"]
    })
    grads <- tape$gradient(top_class_channel, last_conv_layer_output)
    pooled_grads <- mean(grads, axis = c(1, 2, 3), keepdims = TRUE)
    heatmap <-
      (last_conv_layer_output * pooled_grads) %>%
      mean(axis = -1) %>%
      .[1, , ]
  })
  
  output$gradcam <- renderPlot({
    heatmap <- grad_cam_process()
    pal <- hcl.colors(256, palette = "Spectral", alpha = .4, rev = TRUE)
    heatmap <- as.array(heatmap)
    heatmap[] <- pal[cut(heatmap, 256)]
    heatmap <- as.raster(heatmap)
    img <- tf_read_image(img_path, resize = NULL)
    display_image_tensor(img)
    rasterImage(heatmap, 0, 0, ncol(img), nrow(img), interpolate = FALSE)
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
                       monitor = "val_loss"
                     )
                     history <- model %>% fit_generator(
                       generator = training_image_flow, 
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
      test_flow <-  image_flow(input$filedir,"/test/", 
                               10,
                               c(224, 224))
      d <- test_model %>%
        predict(test_flow, steps = test_flow$n/test_flow$batch_size) 
      if (i == 1){
        result <- data.frame(classes = rep(cla[i],length(d[,i])), prob = d[,2]) 
      }else{
        result <- data.frame(classes = rep(cla[i],length(d[,i])), prob = 1-d[,2]) 
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
    test_flow <-  image_flow(input$filedir,"/test/", 
                             10,
                             c(224, 224))
    result <- evaluate(test_model,test_flow,steps = test_flow$n/test_flow$batch_size)
    cat(sprintf("Test accuracy: %.3f\n", result["accuracy"]))
  })
  
  output$detail_model <- renderPrint({
    summary(models())
  })
  
  output$detail_freeze_model <- renderPrint({
    summary(mod())
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
