import get_input
import platform
import matplotlib.pyplot as plt

if platform.system() == 'Windows':
    DATA_PATH = 'D:\Desktop\DEMO_FILE\cifar-10-batches-py'
elif platform.system() == 'Linux':
    DATA_PATH = ' '
else:
    exit('ZBH: unknown platform system %s.' % platform.system())

test, train = get_input.get_cifar10_input(DATA_PATH)

print(test.keys(),test[b'batch_label'])
print(train.keys(),train[b'batch_label'])

plt.imshow(train[b'data'][500])
plt.show()




# np.reshape(a[0][b'data'],[10000,32,32,3],order='F')[0]

