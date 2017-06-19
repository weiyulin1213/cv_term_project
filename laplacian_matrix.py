import numpy as np


def pad_zero(content_img):
	return np.lib.pad(content_img, ((0,0),(1,1),(1,1),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0),(0,0)))

def build_mean(content_img):
	_,h,w,d=content_img.shape
	mean_matrix=np.empty([1,h,w,d])
	content_img=pad_zero(content_img)
	for x in range(h):
		for y in range(w):
			mean_matrix[0,x,y]=np.mean(content_img[0,(x-1+1):(x+1+1),(y-1+1):(y+1+1),:], (0,1,2))
	return mean_matrix
			
def build_cov(content_img):
	_,h,w,d=content_img.shape
	cov_matrix=np.empty([h,w,3,3])
	content_img=pad_zero(content_img)
	for x in range(h):
		for y in range(w):
			sigma=np.array([content_img[0, x-1,y-1,:], content_img[0, x,y-1,:], content_img[0, x+1,y-1,:], \
							content_img[0, x-1,y,:], content_img[0, x,y,:], content_img[0, x+1,y,:], \
							content_img[0, x-1,y+1,:], content_img[0, x,y+1,:], content_img[0, x+1,y+1,:]])
			cov_matrix[ x,y]=np.cov(sigma, bias=True, rowvar=False) / 9.0
	return cov_matrix

def D(i,j,width,height,content_img,mean,cov_matrix):
	if i>j:
		tmp=j
		j=i
		i=tmp
	
	rowi = int(np.floor(i/width))
	coli = int(i % width)
	rowj = int(np.floor(j/width))
	colj = int(j % width)
	if rowi>=height or coli>=width or rowj>=height or colj>=width:
		return 0
	delta = 0
	Wk_size = 9 
	epsilon = 1

	if i==j:
		delta = 1
	
	return delta - 1./Wk_size * (1+np.matmul(np.matmul((content_img[0,rowi,coli] -mean[0,rowi,coli] ).T , np.linalg.inv((cov_matrix[rowi,coli,:,:] + epsilon*np.eye(3)))) , (content_img[0,rowj,colj]-mean[0,rowi,coli])))

def laplacian_col(input_col, w, h, content_img, mean_matrix, cov_matrix):
	col=np.zeros([ w*h])
	j=input_col
	for i in range(w*h):
		if i<=j+1 and i>=j-1:
			col[ i]=D(i-1, j-1, w, h, content_img, mean_matrix, cov_matrix)+D(i-1, j, w, h, content_img, mean_matrix, cov_matrix)+D(i-1, j+1, w, h, content_img, mean_matrix, cov_matrix)+\
					D(i, j-1, w, h, content_img, mean_matrix, cov_matrix)+D(i, j, w, h, content_img, mean_matrix, cov_matrix)+D(i, j+1, w, h, content_img, mean_matrix, cov_matrix)+\
					D(i+1, j-1, w, h, content_img, mean_matrix, cov_matrix)+D(i+1, j, w, h, content_img, mean_matrix, cov_matrix)+D(i+1, j+1, w, h, content_img, mean_matrix, cov_matrix)
	return col 
	

#img=cv2.imread("grid.jpg")
#img=cv2.resize(img, (10,10))

#h,w,d=img.shape
#mean_matrix=build_mean(img)

#cv2.imshow("content image", img)
#cv2.imshow("mean", mean_matrix/255)
#cv2.waitKey()

#print laplacian_col(5,w,h,img,mean_matrix,cov_matrix)
#print laplacian_col(6,w,h,img,mean_matrix,cov_matrix)
