

preprocess_input <- list(
  imagenet_preprocess_input = keras::imagenet_preprocess_input,
  xception_preprocess_input = keras::xception_preprocess_input,
  resnet_v2_preprocess_input = keras::resnet_v2_preprocess_input,
  resnet_preprocess_input = keras::resnet_preprocess_input,
  densenet_preprocess_input = keras::densenet_preprocess_input,
  nasnet_preprocess_input = keras::nasnet_preprocess_input,
  mobilenet_preprocess_input = keras::mobilenet_preprocess_input,
  mobilenet_v2_preprocess_input = keras::mobilenet_v2_preprocess_input,
  xception_preprocess_input = keras::xception_preprocess_input,
  inception_v3_preprocess_input = keras::inception_v3_preprocess_input,
  inception_resnet_v2_preprocess_input = keras::inception_resnet_v2_preprocess_input
)

preprocess_input_list <- names(preprocess_input)


