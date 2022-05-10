import model_zoo
import Mini_Imagenet
import paddle
import os

n_way, n_support, n_query = 5, 1, 16
num_workers, numepochs = 0, 3
encoder_statedict = paddle.load(os.path.join('./save', 'baselinepp_50_model'))
testdataset_1shot = Mini_Imagenet.metaDataSet('test',imagepath = './images')
testdatasetsamplerr_1shot = Mini_Imagenet.BatchSampler('test', n_way, n_support, n_query, len=600, ep_per_batch=1)
test_loader_1shot = paddle.io.DataLoader(testdataset_1shot, batch_sampler=testdatasetsamplerr_1shot,
                                         num_workers=num_workers, use_shared_memory=True)
model_zoo.baselinepp_test(test_loader_1shot, encoder_statedict, numepochs, n_way=n_way, n_support=n_support,
                                    n_query=n_query)

testdataset_5shot = Mini_Imagenet.metaDataSet('test',imagepath = './images')
testdatasetsamplerr_1shot = Mini_Imagenet.BatchSampler('test', n_way, 5, n_query, len=600, ep_per_batch=1)
test_loader_1shot = paddle.io.DataLoader(testdataset_1shot, batch_sampler=testdatasetsamplerr_1shot,
                                         num_workers=num_workers, use_shared_memory=True)
model_zoo.baselinepp_test(test_loader_1shot, encoder_statedict, numepochs, n_way=n_way, n_support=5,
                                    n_query=n_query)
