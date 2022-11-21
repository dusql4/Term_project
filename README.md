# Term Project (Calculation after numeric image recognition) 
***Project team member : 201832607 김연비, 202234904 신연수, 202234949 홍민석, 황반푹***

### Table of Contents
---
1.  Project use
2.  Example Image
3.  Package Used
4.  How to execute
5.  Reference Materials

---
---
##### **1. Project use**
- **Progress in the big direction** : The project proceeds with the flow of recognizing numbers and calculating recognized numbers.

- **Reason for selecting the topic** : The project that recognizes numbers and calculates them has more direction to develop in the future, and it is also selected as the topic because it is considered to be a strength that deep learning data is created and entered.

- **Project details**: input data, convolution layer, poly-connected layer
    \- Input data : The data used as input are handwritten numbers.

    \- Convulution layer : A total of two convolutional layers are used to extract features using convolutional filters in each layer, apply ReLu as an activation function, and summarize key features using Max Pooling.
    
    \- Poly connected layer : Features extracted through convolutional filters are classified by a fully connected layer, and the pully connected layer uses one neural network and uses a dropout layer to prevent overfitting.
    
---
---
##### **2. Example image**

First, Upload a numeric image and find the boundary value as shown in the image below
![이미지2](https://user-images.githubusercontent.com/112548680/202973975-462098ab-858b-4add-b941-544ad93dbefb.jpg)

Second, Find the boundaries of each of the numeric images as shown in the image below.
![이미지1](https://user-images.githubusercontent.com/112548680/202973957-7f978732-1aca-46ca-9894-f6e71d98bf32.jpg)

Third, When each boundary is expressed in numbers, it is output as follows.  
![이미지3](https://user-images.githubusercontent.com/112548680/202973989-95153efd-af63-41d3-a345-f4f402052f05.jpg)

Fourth, The image below is an image that shows each boundary easily.
(It appears as the boundary of the whole, the boundary of the seat of white, the boundary of the seat of ten, and the boundary of the seat of work.)  
<img src="https://user-images.githubusercontent.com/112548680/202974000-7dcaa3f9-7e84-4adb-9c18-274ed1877abb.jpg"  width="500" height="400">

Fifth, It can be seen that the image is well recognized as follows.  
![이미지5](https://user-images.githubusercontent.com/112548680/202974012-8cabe3b8-a2c2-4e61-8108-e574d31ee6d3.png)

---
---
##### **3. Package Used**
- **Mnist** : A large database of handwritten numbers
- **Matplotlib** : Utilize Python programming languages to draw libraries, graphs, and visual materials for static, animation, and interactive visualizations
- **Keras** : Easy and concise deep learning library implemented with Python provides intuitive API
- **Tensorflow** : Google's deep learning library to make it easier for more people to participate in AI development
---
---
##### **4. How to execute**
Let me briefly explain the project execution procedure
1. Upload your own number image
2. Find the boundaries for the whole number
3. Find the boundaries for each numeric image.
4. Recognize each number based on the found boundary.
5. Receive two recognized numbers
6. Receive operator input
7. Calculate two numbers according to the operator
---
---
##### **5. Reference Materials**
```sh
$ Recognizing Numbers : https://kagus2.tistory.com/m/28
$ Related Code: See Open Source Lecture Materials
```
---
