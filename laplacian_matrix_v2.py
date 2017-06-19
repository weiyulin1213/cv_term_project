import numpy as np
import scipy as sp
import tensorflow as tf
import cv2

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
		temp = i
		i = j
		j = temp
	coli = int(np.floor(i/height))
	rowi = int(i % height)
	colj = int(np.floor((j)/height))
	rowj = int(j % height)
	if np.absolute(coli-colj)>1 or np.absolute(rowi-rowj)>1:
		return 0 
			
	if rowi>=height or coli>=width or rowj>=height or colj>=width:
		return 0
	delta = 0
	Wk_size = 9 
	epsilon = 0.001

	if i==j:
		delta = 1
	#print ( (content_img[0,rowi,coli] -mean[0,rowi,coli]).T.shape )
	return delta - 1./Wk_size * (1+np.matmul( np.matmul(np.array ( (content_img[0,rowi,coli] -mean[0,rowi,coli] )).reshape([1,3]) , np.linalg.inv(np.array((cov_matrix[rowi,coli,:,:] + epsilon*np.eye(3))))) , np.array((content_img[0,rowj,colj]-mean[0,rowi,coli]))))

def laplacian_col(input_col, w, h, content_img, mean_matrix, cov_matrix):
	col=np.zeros([ w*h])
	j=input_col
	for i in range(w*h):
		col[i]=D(i, j, w, h, content_img, mean_matrix, cov_matrix)
	return col 
	
def laplacian_sparse_matrix(I , epsilon , win_size):
	neb_size = np.power((win_size*2+1),2)
	_,h,w,c = I.shape
	n = h
	m = w
	img_size = w*h
	indsM = np.arange(img_size).reshape([h, w])
	tlen=img_size*np.power(neb_size,2)

	row_inds=np.zeros([tlen,1])
	col_inds=np.zeros([tlen,1])
	vals=np.zeros([tlen,1])
	_len=0
	for j in range(win_size, w-win_size):
		for i in range(win_size, h-win_size):
			win_inds = indsM[i-win_size:i+win_size+1, j-win_size:j+win_size+1]
			win_inds = win_inds.reshape([neb_size,1])
			winI = I[0,i-win_size:i+win_size+1,j-win_size:j+win_size+1]
			winI = winI.reshape([neb_size,c])
			win_mu = np.mean(winI,0).T
			win_var = np.linalg.inv(np.matmul(winI.T,winI)/neb_size - np.matmul(win_mu,win_mu.T) + epsilon/neb_size*np.eye(c))
			winI= winI - np.matlib.repmat(win_mu.T,neb_size,1)
			tvals = (1+np.matmul(np.matmul(winI,win_var),winI.T))/neb_size
			row_inds[_len:np.power(neb_size,2)+_len]=np.matlib.repmat(win_inds,1,neb_size).reshape([np.power(neb_size,2),1])
			col_inds[_len:np.power(neb_size,2)+_len]=np.matlib.repmat(win_inds.T,neb_size,1).reshape([np.power(neb_size,2),1])
			vals[_len:np.power(neb_size,2)+_len] = tvals.reshape([np.power(neb_size, 2),1])
			_len = _len+np.power(neb_size,2)
	vals=vals[0:_len]
	row_inds=row_inds[0:_len]
	col_inds=col_inds[0:_len]
	indx=np.append(row_inds, col_inds, axis=1)
	#A=tf.SparseTensor(indices=indx, values=tf.cast(vals.reshape([_len,]), tf.float32), dense_shape=[img_size, img_size])
	#sumA=tf.sparse_reduce_sum_sparse(A, axis=1)
	#A=tf.SparseTensor(indices=indx, values=tf.cast(-vals.reshape([_len,]), tf.float32), dense_shape=[img_size, img_size])
	A=sp.sparse.coo_matrix((vals.reshape([_len,]), (row_inds.reshape([_len,]), col_inds.reshape([_len,]))), shape=(img_size, img_size))
	sumA=np.array(A.sum(axis=0)).reshape([img_size,])
	row_diag=np.arange(img_size)
	col_diag=np.arange(img_size)
	sumA_diag=sp.sparse.coo_matrix((sumA, (row_diag, col_diag)), shape=(img_size, img_size))

	L=sumA_diag-A
	new_row_inds=np.array(L.nonzero()[0]).reshape([len(L.nonzero()[0]),1])
	new_col_inds=np.array(L.nonzero()[1]).reshape([len(L.nonzero()[0]),1])
	new_indx=np.append(new_row_inds, new_col_inds, axis=1)
	#result=tf.SparseTensor(indices=tf.cast(new_indx, tf.int64), values=tf.cast(L.data, tf.float32), dense_shape=[img_size, img_size])
	result=np.array(L.toarray())
	np.savetxt("L_result.csv", result, delimiter=",")
	return result

img=cv2.imread("./image_input/lion.jpg")
cv2.imwrite("grid_resize.jpg",cv2.resize(img, (5,5)))
img=cv2.resize(img, (5,5)).reshape([1,5,5,3])
np.savetxt("R.csv", img[0,:,:,0], delimiter=",")
np.savetxt("G.csv", img[0,:,:,1], delimiter=",")
np.savetxt("B.csv", img[0,:,:,2], delimiter=",")

#_,h,w,d=img.shape
#mean_matrix=build_mean(img)
#cov_matrix = build_cov(img)
#cv2.imshow("content image", img)
#cv2.imshow("mean", mean_matrix/255)
#cv2.waitKey()

print (laplacian_sparse_matrix(img,0.0001,1))
#print (laplacian_col(1,w,h,img,mean_matrix,cov_matrix))
