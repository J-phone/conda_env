import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,BatchNormalization,Flatten,Dense,Activation,Input,concatenate,\
MaxPool2D,AveragePooling2D,GlobalAveragePooling2D,add,Reshape
BatchNormalization._USE_V2_BEHAVIOR = False #解決BN層特有問題
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical 
import random 
import math
import functools
import copy
import time
from numba import cuda
import gc

#options nvidia "NVreg_RestrictProfilingToAdminUsers=0"


def clear_fun():
    #cuda.select_device(0)
    #cuda.close()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


#大多core dump問題發生為layer接收到的輸入/輸出有問題
'''
CURRENT_PROBLEMS:
1.loss nan問題仍可能出現 原因仍帶查。 未來是否用call_back方式看loss nan時自動重起gpu及重至計算圖？否則幕前遇到只能重起vscoded

2.輸出的fil_num是否變小一點

*** 一定要在call/build/__init__的參數加上**kwargs，不然會發生超怪問題***
***跑完要用 del model_name 幹掉model物件 並配合tf.keras.backend.clear_session() tf.compat.v1.reset_default_graph()重置計算圖防止首次訓練後loss 無限大(nan)***
***跑完後用來自numba的cuda來 cuda.select_device(0) cuda.close()防止gpu的vram被佔用***
***原本用得regular（he_..）/initializer好像會也會產生loss nan的問題？ 先別用***

0.若首個DPN block的話其inputshape相關參數及input所衍生分支要怎麼處理(特別建立first-block解決)

1.drop out在此種dense/shortcut結構位置，以1*1 layer使輸出配合輸入形狀使resnet發揮作用？

2.在building block中插入多曾以便日後整合SSO(內部short/concate的行可以列表存？）---> 現在把block ban了，每層都有1*1_conv及se_block，
  並以l1的長度來形成概念上的block(cell)，每個層及block間保有add/dense兩條通道，直至最後一個block的末層才合併並輸出
（之前曾以原認定的block為層

2.5 每層間輸入/輸出以DPN型式分別有shortcut 及 densely connect (done)

3.建模型 要隨cell增加而允許特徵深度上升 (暫讓現在的cell數使深度增加，未來讓SSO來？

4. add路徑的形狀一直不變(永遠=1，待解決，日後參考DPN？)

5.有時首個epoch時val_acc會優於train_acc，但後續epoch後似乎就不會了且不影結果

6.caculated_fil_num(SSO的level_2運算結果)被用以把每層的輸入以1*1_conv增加channel，但是在每曾接收到input後才轉的，故對每曾輸入深度及輸出深度不影響

7.輸出深度是由層再cell內的位置及cell數所決定
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)

train,test=tf.keras.datasets.fashion_mnist.load_data()#載入data到ram中
#train,test=tf.keras.datasets.mnist.load_data()
img,label=train #把trainset分派至img / label
timg,tlabel=test
img,timg=img/255.0,timg/255.0 #正規化
img,timg=img[:,:,:,np.newaxis],timg[:,:,:,np.newaxis] #增加color channel維度
label,tlabel=to_categorical(label),to_categorical(tlabel)
label,tlabel=label.astype(np.int32),tlabel.astype(np.int32) #data type轉為int32

vimg,vlabel=img[:10000],label[:10000]
img,label=img[10000:],label[10000:]


#變成dataset
trads=tf.data.Dataset.from_tensor_slices((img,label)) 
trads=trads.shuffle(400).batch(70)  #shuffle的budder_size為500，batch_size=30

vds=tf.data.Dataset.from_tensor_slices((vimg,vlabel))
vds=vds.shuffle(300).batch(70)

teds=tf.data.Dataset.from_tensor_slices((timg,tlabel)) 
teds=teds.shuffle(300).batch(70)


#one kind of building block 
class full_layer(tf.keras.layers.Layer):
    #block num實為每個bloack內層數
    def __init__(self,layer_type,layer_num,calulated_fil_num,cardinality,strides,fil_size,
    sqe_num,layer_conn,total_lat_in_block,cell_num,total_cell_num=None,weight_decay=5e-4,**kwargs):
        super().__init__()
        self.total_cell_num=total_cell_num
        self.cell_num=cell_num
        self.layer_num=layer_num
        self.calulated_fil_num =int(calulated_fil_num*cell_num*0.6) #the sso fil_num determine the input filter_num to 3*3 conv，cell_num*0.8使特徵愈後面的block的層可學到更深的
        self.output_cha=int(self.layer_num*cell_num*1.1) #out_cha_num for each 3*3 cardinality ( total output_channel=int(clayer_num*cardinality*ell_num*0.8)         self.layer_num=layer_num
        self.layer_conn=layer_conn #determine wather of not send the tensor next few layers inside block
        self.layer_type=layer_type
        self.total_lat_in_block=total_lat_in_block
        self.cardinality=cardinality
        self.strides=strides
        self.fil_size=fil_size
        self.weight_decay=weight_decay
        self.BatcNorm,self.BNfir=BatchNormalization(axis=-1),BatchNormalization(axis=-1)
        self.one_by_one_conv=None
        self.add_block=add_block()
        self.width,self.out,self.x_car,self.short_cut,self.dense,self.orig,self.den_orig=None,None,None,None,None,None,None
        self.gruop_list, self.conv_li,self.se_li,self.BN_li,self.rand_li=[],[],[],[],random.sample(list(range(cardinality)),sqe_num)
        self.layer_li=[]
        self.k=0
        self.ll,self.uu=0,0
        for i in range(sqe_num):
            self.se_li.append(sqeeze_excitation(0.7))
        if int(layer_type)==3:
            self.BN_li=[BatchNormalization(axis=-1) for m in range(cardinality)]
            

    def build(self,input_shape,**kwargs):
        #prevent fil_size>input
        self.width=input_shape[0][1] #forece to evaluate the value of variable (scraped)
        self.in_cha=input_shape[0][-1]+input_shape[1][-1] #the number of total input channel
        self.in_cha_short=input_shape[0][-1] #the number of channels of short cut path
        
        print('FULL')
        print('CAR',self.cardinality)
        print('output_cha',self.output_cha)
        print('self.calulated_fil_num',self.calulated_fil_num)
        if self.output_cha*self.cardinality>100:
            self.output_cha=int(100/self.cardinality)
        if self.calulated_fil_num>1000:
            self.calulated_fil_num=1000
        print('output_cha_1',self.output_cha)
        print('self.calulated_fil_num_1',self.calulated_fil_num)
        
        print('Cell_num',self.cell_num,'La_num',self.layer_num,'full_in_channel,Short:',self.in_cha_short,'Dnese',input_shape[1][-1])
        if self.width<self.fil_size:
            self.fil_size=(self.width,self.width)
        if self.width==1: #width=1沒辦法再用pool變小只能卷積
            self.layer_type==1

        #prevent calulated_fil_num<cardinality_num(就算layer_type=2或3，仍會因one_by_one conv把輸入變厚)，層別為conv時有out_shape限制
        #但層別為pool/BN時只能靠控制one_by_one conv限制每層輸出厚度
        if self.calulated_fil_num<self.cardinality:
            if self.layer_type==1:
                self.calulated_fil_num=self.cardinality*max([self.layer_num,self.cell_num])
            else:
                self.calulated_fil_num=min((self.cardinality*max([self.layer_num,self.cell_num])),100)

        self.one_by_one_conv=Conv2D(filters=self.calulated_fil_num,kernel_size=(1,1),strides=1,use_bias=False,
        padding='same')

        self.mpool=MaxPool2D(pool_size=self.fil_size,padding='same',strides=self.strides)
        
        '''
        #prevent output_channel<input channel(但是用concatenate的方式，似乎無必要，因為可能會使每增加的厚度超過200)
        #且out跟in的關係是concatenate上去加厚而非加上
        if self.output_cha*self.cardinality<self.in_cha:
            self.output_cha=int(self.in_cha/self.cardinality)+1 #plus one make sure have channels for dense_path
        '''
        if self.layer_type==1:
            for car in range(self.cardinality):

                self.conv_li.append('c'+str(car))

                self.conv_li[car]=Conv2D(filters=self.output_cha,strides=1,use_bias=False,
                    kernel_size=self.fil_size,padding='same')
        else: 
            self.conv_li=None
        self.u,self.m=divmod(self.calulated_fil_num,self.cardinality)

   
    @tf.function#input_signature=(tf.TensorSpec(shape=[None,None,None,None],dtype=tf.float32),)
    #要放多層再一個block內，要以list把個層前的short cut/dense 們存下?(scraped)
    def call(self,inputs,training=None, mask=None,**kwargs):
        self.gruop_list.clear()
        
        if self.layer_conn==0 and self.cell_num!=self.total_cell_num:
            return [inputs[0],inputs[1]]

        self.orig=inputs[0]
        self.den_orig=inputs[1]
        inputs=concatenate(inputs,axis=-1)
        inputs=self.BNfir(inputs)
        inputs=self.one_by_one_conv(inputs)
        self.k=0

        print('OUT—CHA',self.output_cha*self.cardinality)
        for car in range(self.cardinality):
            self.ll=car if car<=self.m else self.m
            self.uu=car+1 if car+1<=self.m else self.m
            if self.layer_type==1:
                self.x_car=self.conv_li[car](inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            if self.layer_type==2:         
                self.x_car=self.mpool(inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            if self.layer_type==3:
                self.x_car=self.BN_li[car](inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            
            if car in self.rand_li:
                self.x_car=self.se_li[self.k](self.x_car)
                #print(self.se_li[self.k].trainable_variables)
                self.k+=1
            self.gruop_list.append(self.x_car)
           

        self.out=concatenate(self.gruop_list,axis=-1)
        self.out=self.BatcNorm(self.out)
        self.out=Activation('relu')(self.out)
        if self.layer_type==2:
            self.orig=MaxPool2D(pool_size=self.fil_size,padding='same',strides=self.strides)(self.orig)
            self.den_orig=MaxPool2D(pool_size=self.fil_size,padding='same',strides=self.strides)(self.den_orig)

        self.short_cut=self.add_block([self.orig,self.out[:,:,:,:self.in_cha_short]])
        self.dense=concatenate([self.den_orig,self.out[:,:,:,self.in_cha_short:]]) 
        del self.out
        if (self.total_cell_num==self.cell_num) and (self.layer_num==self.total_lat_in_block):  #末個cell的末個layer再合
            print('FFFFFIIIIIIIIIIIIINNNNNAAAAAAAAAALLLLL')
            print('BEFOR_CONCATE',self.short_cut,self.dense)
            print("CONCATE",concatenate([self.short_cut,self.dense],axis=-1))
            return concatenate([self.short_cut,self.dense],axis=-1)
        else: 
            if self.layer_conn==3: #connect topology=3 回傳short/dense
                print('33333333333333')
                print('TSHORT',self.short_cut,'TDENSE',self.dense)
                return [self.short_cut,self.dense]        
            elif self.layer_conn==2:#connect topology=2 只回傳dense
                return [self.orig,self.dense]
            else:#connect topology=1 只回傳shortcut
                print('1111111111111')
                return [self.short_cut,self.den_orig]


class first_layer(full_layer):

    def __init__(self,layer_type,calulated_fil_num,cardinality,strides,incr,\
    sqe_num,fil_size,weight_decay=5e-4,layer_conn=3,total_lat_in_block=1,layer_num=1,**kwargs):
        print('FIR_LAYER_INIT')
        super().__init__(layer_type,layer_num,calulated_fil_num,cardinality,strides,fil_size,
        sqe_num,layer_conn,total_lat_in_block,cell_num=1,**kwargs)
        self.output_cha=3
        self.calulated_fil_num=calulated_fil_num
        self.incr=incr
        if layer_type==2:
            self.mpool=MaxPool2D(pool_size=fil_size,padding='same')
        self.orig,self.den_orig,self.x_car,self.out=None,None,None,None
        self.activa=Activation('relu')
        self.x=0
        

    def build(self,input_shape,**kwargs):
        self.width=input_shape[1]
        self.cha=input_shape[-1]
        print('FIRST',input_shape)

        #inp_shape control
        if self.calulated_fil_num<self.cha+self.incr:
            self.calulated_fil_num=self.cha+self.incr

        self.one_by_one_conv=Conv2D(filters=self.calulated_fil_num,kernel_size=(1,1),strides=1,use_bias=False,
        padding='same')

        if self.output_cha*self.cardinality<self.cha+self.incr:
            self.output_cha=math.ceil(self.cha/self.cardinality)+math.ceil(self.incr/self.cardinality)
        
        for car in range(self.cardinality):

            self.conv_li.append('c'+str(car))

            self.conv_li[car]=Conv2D(filters=self.output_cha,strides=1,use_bias=False,
                    kernel_size=self.fil_size,padding='same')
        self.u,self.m=divmod(self.calulated_fil_num,self.cardinality)

    @tf.function    
    def call(self,inputs,training=None, mask=None,**kwargs):
        self.gruop_list.clear()
        inputs=self.BNfir(inputs)
        inputs=self.one_by_one_conv(inputs)
        self.orig,self.den_orig=inputs[:,:,:,:self.cha],inputs[:,:,:,self.cha:]

        self.k=0
        for car in range(self.cardinality):
            self.ll=car if car<=self.m else self.m
            self.uu=car+1 if car+1<=self.m else self.m
            if self.layer_type==1:
                self.x_car=self.conv_li[car](inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            if self.layer_type==2:               
                self.x_car=self.mpool(inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            if self.layer_type==3:
                self.x_car=self.BN_li[car](inputs[:,:,:,car*self.u+self.ll:(car+1)*self.u+self.uu])
            if car in self.rand_li:
                self.x_car=self.se_li[self.k](self.x_car)
                self.k+=1
            self.gruop_list.append(self.x_car)
        self.out=concatenate(self.gruop_list,axis=-1)
        self.out=self.BatcNorm(self.out)
        self.out=self.activa(self.out)
        if self.layer_type==2:
            self.orig=self.mpool(self.orig)
            self.den_orig=self.mpool(self.den_orig)

        self.short_cut=self.add_block([self.orig,self.out[:,:,:,:self.cha]])
        self.dense=concatenate([self.den_orig,self.out[:,:,:,self.cha:]],axis=-1)
        print('FIR OUT',[self.short_cut, self.dense])
        return [self.short_cut, self.dense]


class sqeeze_excitation(tf.keras.layers.Layer):
    
    def __init__(self,ratio,**kwargs):
        super().__init__(**kwargs)
        self.x1=None
        self.out_cha=None
        self.ratio=ratio
        self.pool=GlobalAveragePooling2D()
        self.relu=Activation('relu')
        self.sig=Activation('sigmoid')        

    def build(self,inputs_shape):
        self.out_cha=inputs_shape[-1]
        self.dense1,self.dense2=Dense(int(self.out_cha*self.ratio)),Dense(inputs_shape[-1])
        self.reshape=Reshape((1,1,self.out_cha))

    @tf.function
    def call(self,inputs,**kwargs):
        self.x1=self.pool(inputs)
        self.x1=self.dense1(self.x1)
        self.x1=self.relu(self.x1)
        self.x1=self.dense2(self.x1)
        self.x1=self.sig(self.x1)
        self.x1=self.reshape(self.x1)
        self.x1=self.x1*inputs

        return self.x1

class add_block(tf.keras.layers.Layer):
   
    def __init__(self,subsample=False,**kwargs):
        super().__init__(**kwargs)
        self.subsample=subsample
        self.orginal_inp,self.dense_inp=None,None
        self.pool=MaxPool2D(padding='same',pool_size=(2,2))
        self.out=None
    
    def build(self,inputs_shape,**kwargs):
        self.original_cha=inputs_shape[0][-1]
        self.dense_cha=inputs_shape[1][-1]
        
    @tf.function
    def call(self,inputs,**kwargs):
        self.orginal_inp,self.dense_inp=inputs[0],inputs[1]

        if self.subsample:
            self.orginal_inp=self.pool(padding='same',pool_size=(2,2))
            
        if self.original_cha!=self.dense_cha:
            raise StopIteration
        self.out=add([self.orginal_inp,self.dense_inp])
        return self.out

class build_block(tf.keras.layers.Layer):
    
    def __init__(self,l1,l2,l3,current_cell,total_cell_num,**kwargs):
        super().__init__(**kwargs)

        self.block_li=[]
        self.total_layer_num=int((len(l1)-1)/3) #為每個block內的總層數

        l2_ind,l3_ind=0,0
        #迭帶所有block內的layer
        for curr_la_num,inde1 in enumerate(range(1,len(l1),3),1):
            print('CELL',current_cell)
            print('LAYER',curr_la_num)
            self.x=None
            car_num=functools.reduce(self.merge,l1[inde1+1:inde1+3])
            fil_num=functools.reduce(self.merge,l2[l2_ind+1:l2_ind+4]) if int(l1[inde1])==1 else 0
            se=functools.reduce(self.merge,l3[l3_ind:l3_ind+2])    
            #init block內的每層
            self.layer_in_block=\
            full_layer(layer_type=int(l1[inde1]),\
            calulated_fil_num=int(fil_num),\
            cardinality=int(car_num),strides=1,\
            fil_size=int(l2[l2_ind]),weight_decay=5e-4,\
            sqe_num=int(se),\
            total_lat_in_block=self.total_layer_num,\
            cell_num=current_cell,\
            layer_conn=int(l2[l2_ind+4]),\
            layer_num=curr_la_num,\
            total_cell_num=total_cell_num)
            self.block_li.append(self.layer_in_block)
            l2_ind+=5
            l3_ind+=2
            tf.print('HOLE_BLOCK',self.block_li)
            
    def build(self,input_shape):
        print('BUILD_BLOCK_INPUT',input_shape)

    
    def merge(self,ele1,ele2):
        return str(int(ele1))+str(int(ele2))
     
       
    @tf.function
    def call(self,inputs,**kwargs):
        if self.x==None:
            self.x=self.block_li[0](inputs)
    
        for j in range(1,len(self.block_li)):
            self.x=self.block_li[j](self.x)

        return self.x


class full_model(tf.keras.Model):

    def __init__(self,l1,l2,l3,**kwargs):
        super().__init__(**kwargs)
        #self.total_cell_num=l1[0]
        total_cell_num=l1[0]
        self.full_net=[]
        self.xb=None
        #init first layer
        self.fir=first_layer(layer_type=1,\
        calulated_fil_num=100,\
        incr=60,\
        sqe_num=30,\
        cardinality=30,strides=1,\
        fil_size=(3,3),weight_decay=5e-4,\
        layer_conn=3)
#--------------------以layer達到類似效果用------------------ 
        self.t=[]
        self.block_li=[]
        self.total_layer_num=int((len(l1)-1)/3) #為每個block內的總層數
        m=0
        #迭帶所有block內的layer

        for current_cell in range(total_cell_num):
            l2_ind,l3_ind=0,0
            for curr_la_num,inde1 in enumerate(range(1,len(l1),3),1):
                print('CELL',current_cell)
                print('LAYER',curr_la_num)
                self.x=None
                car_num=functools.reduce(self.merge,l1[inde1+1:inde1+3])
                fil_num=functools.reduce(self.merge,l2[l2_ind+1:l2_ind+4]) if int(l1[inde1])==1 else 0
                print('TTfil_num',fil_num)
                se=functools.reduce(self.merge,l3[l3_ind:l3_ind+2])    
                #init block內的每層
                self.block_li.append('cl'+str(m))
                self.block_li[m]=\
                full_layer(layer_type=int(l1[inde1]),\
                calulated_fil_num=int(fil_num),\
                cardinality=int(car_num),strides=2,\
                fil_size=int(l2[l2_ind]),weight_decay=5e-4,\
                sqe_num=int(se),\
                total_lat_in_block=self.total_layer_num,\
                cell_num=int(current_cell+1),\
                layer_conn=int(l2[l2_ind+4]),\
                layer_num=curr_la_num,\
                total_cell_num=total_cell_num)

                l2_ind+=5
                l3_ind+=2
                m+=1


        #init the blocks
        '''
        for cel in range(self.total_cell_num):
            self.full_net.append(build_block(l1,l2,l3,cel+1,self.total_cell_num))
            print('FULLNET',self.full_net)
        '''
        self.fla=Flatten()
        self.out_dense=Dense(10,activation='softmax')

        out=[self.fla,self.out_dense];self.full_net.extend(out)
        self.t.append(self.fir)
        self.t.extend(self.block_li)
        self.t.extend(out)

    def merge(self,ele1,ele2):
        return str(int(ele1))+str(int(ele2))
    
    def save_layer(self,layer_li):
        return layer_li

    #@tf.function
    def call(self,inputs,**kwargs):
        for ind,m in enumerate(self.t):
            if ind==0:
                self.xb=m(inputs)
            else:
                self.xb=m(self.xb)

        #test under layer
        return self.xb

    def __del__(self):
        print('Model Deleted')


def call_full_model(l1,l2,l3):
    
    print(l1,l2,l3)
    clear_fun()
    gc.collect()
    md=full_model(l1,l2,l3)
    #tf.summary.trace_on(graph=True,profiler=True) 
    md.compile(optimizer=tf.keras.optimizers.SGD(decay=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])#要debug可加參數:run_eagerly=True，否則影響效能
    md.fit(trads,validation_data=vds,epochs=2)#,callbacks=[tensorboard_callback]
    md.summary()
    teloss,teacc=md.evaluate(teds)
    res=copy.deepcopy(teacc)
    del md.block_li
    del md
    gc.collect()
    clear_fun()
    gc.collect()
    return res


    



if __name__=="__main__":
    for j in range(100):
        c=call_full_model([3,2,2,2,1,2,2],[1,0,9,9,1,1,0,3,3,0],[2,1,2,0])
        print(j)

