import cv2 as cv
import numpy as np
import glob
SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
merged=[]
labels=[]
no_of_samples=1000
no_of_classes=34
classes={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'J',19:'K',20:'L',21:'M',22:'N',23:'P',24:'Q',25:'R',26:'S',27:'T',28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z'}#
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
def extract(img):
    #buf=[]
    #for row in range(0,img.shape[0]):
     #   for col in range(0,img.shape[1]):
      #      buf.append(img[row][col])
    buf=np.array(img)
    buf=buf.reshape(1,900)
    norm=((buf-buf.min())/(buf.max()-buf.min()))
    norm=list(norm[:])
    print(norm)
    #norm=[(pix-min(buf))/float(max(buf)-min(buf)) for pix in buf]
    merged.append(norm)
        

for i in range(0,34):
    ct=0
    
    for path in glob.glob("data/{}/*.jpg".format(classes[i])):
        #print("path",path,"ct",ct)
        if ct>=no_of_samples:break
        image=cv.imread(path)
        image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        deskew_image=image#deskew(image)
        resize_image=cv.resize(deskew_image,(30,30))
        #ret,resize_image=cv.threshold(resize_image, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)
        extract(resize_image)
        labels.append(i)
        ct+=1
    print("Ct ,i :",ct,i)
#print("t",type(merged),len(merged),'\n',merged)
traindata=np.float32(merged)
responses=np.array(labels).reshape(34*no_of_samples,1)
svm.train(traindata, cv.ml.ROW_SAMPLE, responses)
svm.save('toloadmodels/svm_data_withoutdeskew_20.dat')
