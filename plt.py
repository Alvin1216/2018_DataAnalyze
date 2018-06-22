import numpy as np
import matplotlib.pyplot as plt

# Please create a 200x300 numpy array based on Binomial random sample.
data=np.random.binomial(2,0.5,(200,300))
#Calculate the mean and standard deviation for each columns as a new one-dimensional array.
for a in range(200):
    std_col=np.std(data[a])
    mean_col=np.mean(data[a])
    print("Standard Deviation in col "+str(a+1)+": "+str(std_col))
    print("Mean in col "+str(a+1)+": "+str(mean_col))

# Sort the one-dimensional array
#the array has 10 random integer number
sorting_array=np.random.randint(10,size=(1,10))
print("Before sorting: "+str(sorting_array))
print("After sorting: "+str(np.sort(sorting_array)))

#Reshape the one-dimensional array into a 30x10 2D array
#creat a one dimension array and reshape to 30*10 2d array
array=np.random.randint(10,size=300)
print("Before reshaping: "+str(array))
reshaped_array=np.reshape(array, (30,10))
print("After reshaping: ")
for p in range(30):
    print(reshaped_array[p]);

#Extract the 2nd, 4th columns.
print("column2:"+str(reshaped_array[2-1]))
print("column4:"+str(reshaped_array[4-1]))

# plot the sequences for 2nd, 4th columns, separately.
plt.plot(reshaped_array[2-1] ,'b',label="2nd")
plt.plot(reshaped_array[4-1] ,'r',label="4th")
plt.ylabel('values')
plt.xlabel('array_index')
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
