import model_zoo
import paddle
import Mini_Imagenet
import paddle.optimizer as pdoptimizer

model = model_zoo.get_baselinepp_model()
pretraindataset = Mini_Imagenet.classifyDataSet(imagepath = './images')
base_loader = paddle.io.DataLoader(pretraindataset, batch_size=16, shuffle=True, num_workers=14)
optimizer = pdoptimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
model = model_zoo.baselinepp_train(base_loader, model, optimizer, start_epoch=0, stop_epoch=50, save_freq=5)
