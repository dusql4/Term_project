import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def img2int(img_root):
    #이미지 읽어오기
    img = cv2.imread(img_root)
    plt.figure(figsize=(15,12))


    #이미지 흑백처리
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 

    #이미지 블러
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #이미지 내의 경계 찾기
    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy= cv2.findContours(img_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #경계를 직사각형으로 찾기
    rects = [cv2.boundingRect(each) for each in contours]

    #왼쪽에 있는 경계 순서대로 정렬
    rects=sorted(rects)
    thickness=abs(rects[0][2]-rects[1][2])*2

    #가장 밖에 있는 경계선 찾기
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    #찾은 경계선 흰색으로 칠하기
    cv2.drawContours(img_blur, biggest_contour,-1,(255,255,255),thickness)


    #경계선 지우고 경계 다시 찾기 : 숫자만 찾기 위해서
    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy= cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #경계 직사각형 찾기
    rects = [cv2.boundingRect(each) for each in contours]
    #왼쪽부터 읽어와야 하므로 정렬 
    rects=sorted(rects)

    # 사각형 영역 추출 확인하기
    # for rect in rects:
    #     print(rect)
    #     cv2.circle(img_blur, (rect[0],rect[1]),10,(0,0,255), -1)
    #     cv2.circle(img_blur, (rect[0]+rect[2],rect[1]+rect[3]),10,(0,0,255), -1)
    #     cv2.rectangle(img_blur,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),3)
        



    #이전에 처리해놓은 이미지 사용
    img_for_class = img_blur.copy()

    #최종 이미지 파일용 배열
    mnist_imgs=[]
    margin_pixel = 15

    #숫자 영역 추출 및 (28,28,1) reshape

    for rect in rects:
        print(rect)
        #숫자영역 추출
        im=img_for_class[rect[1]-margin_pixel:rect[1]+rect[3]+margin_pixel,rect[0]-margin_pixel:rect[0]+rect[2]+margin_pixel]
        row, col = im.shape[:2]
        
        #정방형 비율을 맞춰주기 위해 변수 이용
        bordersize= max(row,col)
        diff=min(row,col)
        
        #이미지의 intensity의 평균을 구함
        bottom = im[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        # border추가해 정방형 비율로 보정
        border = cv2.copyMakeBorder(
            im,
            top=0,
            bottom=0,
            left=int((bordersize-diff)/2),
            right=int((bordersize-diff)/2),
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )
        
        
        square=border
    
        
        #square 사이즈 (28,28)로 축소
        resized_img=cv2.resize(square,dsize=(28,28),interpolation=cv2.INTER_AREA)
        mnist_imgs.append(resized_img)
    
        
    result=0
    model = tf.keras.models.load_model("mnist_model")
    for i in range(len(mnist_imgs)):

        img = mnist_imgs[i]
        # 이미지를 784개 흑백 픽셀로 사이즈 변환
        img=img.reshape(-1, 28, 28, 1)
    

        # 데이터를 모델에 적용할 수 있도록 가공
        input_data = ((np.array(img) / 255) - 1) * -1
        input_data

        # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
        res = np.argmax(model.predict(input_data), axis=-1)

        
        result*=10
        result+=res[0]
    return result

