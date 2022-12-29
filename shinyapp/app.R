#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
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

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("The Great Deep-Learning Panel"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
          textInput("filedir",
                    "input your task path",
                    "dataset"),
          sliderInput("dropout_factor", 
                      "Set a probability for dropout layer",
                      min = 0,
                      max = 1,
                      value = 0.2),
            sliderInput("epoch",
                        "Number of epoch",
                        min = 1,
                        max = 100,
                        value = 3),
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
              "测试结果", plotOutput("predict")
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
    length(classes)
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
  
  test_flow <- reactive({
    flow_images_from_directory(
      generator = validation_image_gen(),
      directory = paste0(getwd(), "/", input$filedir, "/test/"), 
      target_size = c(224, 224),
      class_mode = NULL,
      shuffle = FALSE
    )
  }) 
  
  models <- reactive({
    nums <- classnum()
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
    models <- model()
    withProgress(message = 'Calculation in progress',
                 detail = 'This may take a while...', value = 0, {
    model %>% 
      compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = "accuracy")
    # Include the epoch in the file name
    checkpoint_path <- paste0(getwd(),"/",input$model, 
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

  
  output$predict <- renderPrint({
    test_flow <- test_flow()
    traincnn() %>% predict_generator(
        test_flow,
        steps = test_flow$n/test_flow$batch_size
      )
      
  })
 
 
}

# Run the application 
shinyApp(ui = ui, server = server)
