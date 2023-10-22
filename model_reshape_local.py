from tensorflow import keras
import sys

# 从命令行参数获取h5文件的路径
h5_file_path = sys.argv[1]

# 导入原始模型（包括分类层）
original_model = keras.models.load_model(h5_file_path)

# 创建一个新的模型，去除原始模型的最后一层
new_model = keras.models.Model(inputs=original_model.input, outputs=original_model.layers[-2].output)

# 保存新模型
new_model.save('new_model.keras')
