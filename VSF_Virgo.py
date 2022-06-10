import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#added from stropy.io.fits.hdu.hdulist import _File, FILE_MODES
from astropy.io.fits.hdu.hdulist import _File, FILE_MODES
from astropy.io import fits
from matplotlib.colors import LogNorm
import scipy
from scipy import ndimage as ndi
from scipy.stats import norm
from multiprocessing import Pool
import random


cut=False

newparams = {'axes.labelsize': 16, 'axes.linewidth': 1, 'savefig.dpi': 300,
             'lines.linewidth': 1.5, 'figure.figsize': (8, 6),
             'figure.subplot.wspace': 0.4,
             'ytick.labelsize': 14, 'xtick.labelsize': 14,
             'ytick.major.pad': 5, 'xtick.major.pad': 5,
             'legend.fontsize': 16, 'legend.frameon': True,
             'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

#added ignore_missing_simple=True
ignore_missing_simple=True

velo = fits.open('M87_velocitymap.fits')
velo_data = velo[0].data
velo.close()

error=fits.open("M87_errvelocitymap.fits")
error_data = error[0].data
error.close()

res=0.0159 #pixel size in kpc.
max_error=20. #max velocity error in km/s.

(Cx,Cy)=(155, 153.) #position of SMBH

#added make_mask function
#create a mask to select a fraction of the points.
def make_mask(velo_data=velo_data,mask_size=100000):
  a = np.zeros(velo_data.shape, dtype=int)
  b=a.flatten()
  b[0:mask_size] = 1
  np.random.shuffle(b)
  b = b.astype(bool)
  a=b.reshape(velo_data.shape)
  return a

def make_data(max_error,velo_data=velo_data,error_data=error_data):
  xvalues = np.arange(0,len(velo_data[0,:]))
  yvalues = np.arange(0,len(velo_data[:,0]))
  xx, yy = np.meshgrid(xvalues, yvalues)

  #added this_mask
  #add this right after the function definition of make_data
  this_mask=make_mask(velo_data=velo_data,mask_size=1000)

  #this is an experiment similar to jackknife resampling. Default is using the full image (no cut).
  if cut is True:
    slope=np.arctan(np.pi/2)
    random_mask=(yy>Cy)&(yy<slope*(xx-Cx)+Cy)
    good_v=(velo_data<1000)&(velo_data>-2000)&(error_data<max_error)&random_mask
  else:
    good_v = (velo_data < 1000) & (velo_data > -2000) & (error_data < max_error) & this_mask
    print("good_v",good_v.shape)
    print("total number of points used:", velo_data[good_v].shape)
    velo_plot = np.ma.masked_array(velo_data, mask=~good_v)
    #first make a velocity map
    plt.clf()
    plt.plot(Cx,Cy,marker="X",color="k",markersize=10,linestyle="None",label="Black Hole Position")
    plt.imshow(velo_plot,vmin=-300, vmax=300,cmap="bwr")
    y_labels=np.arange(0,4)
    y_locs=y_labels/res
    x_labels=np.arange(0,4)
    x_locs=x_labels/res
    plt.xticks(x_locs, x_labels)
    plt.yticks(y_locs, y_labels)
    plt.xlabel("x (kpc)")
    plt.ylabel("y (kpc)")
    plt.gca().invert_yaxis()
    cb=plt.colorbar()
    cb.set_label("line-of-sight velocity (km/s)")
    plt.legend(loc="upper left",prop={'size': 14})
    plt.savefig("Virgo_Halpha_v_use.png")
    #begin calculations of l and delta_v
    goodxx=xx[good_v]
    goodyy=yy[good_v]
    vel_a = np.reshape(velo_data[good_v], (velo_data[good_v].size, 1))
    vel_b = np.reshape(velo_data[good_v], (1, velo_data[good_v].size))
    v_diff_matrix = vel_a - vel_b
    error_a = np.reshape(error_data[good_v]**2, (error_data[good_v].size, 1))
    error_b = np.reshape(error_data[good_v]**2, (1, error_data[good_v].size))
    error_matrix = error_a + error_b
    px_a = np.reshape(goodxx, (goodxx.size, 1))
    px_b = np.reshape(goodxx, (1, goodxx.size))
    py_a = np.reshape(goodyy, (goodyy.size, 1))
    py_b = np.reshape(goodyy, (1, goodyy.size))
    dist_matrix = np.sqrt((px_a - px_b)**2 + (py_a - py_b)**2)
    v_diff_half = np.ndarray.flatten(np.triu(v_diff_matrix, k=0))
    dist_half = np.ndarray.flatten(np.triu(dist_matrix, k=0)) # this is still a 2D matrix, with the lower half values all set to 0
    error_half=np.ndarray.flatten(np.triu(error_matrix, k=0))
    good_dist = dist_half>0
    np_dist=dist_half[good_dist]
    np_v_diff=v_diff_half[good_dist]
    np_error_2=error_half[good_dist]
    #np.savez("M87_save.npz",np_dist=np_dist,np_v_diff=np_v_diff,np_error_2=np_error_2)
    return np_dist, np_v_diff, np_error_2


def make_vsf_plot(d_max=600,n_bins=200,check_Guassian=False):
  (np_dist,np_v_diff, np_error_2)=make_data(max_error=20)
  #need to be careful with creating the list of l. The first few values need to be set by hand because of the discretization of the map.
  dist_floor=np.floor(np_dist*100)
  unique=np.unique(dist_floor)/100.
  dist_array=np.append(unique[0:15],np.logspace(np.log10(unique[15]),np.log10(d_max),n_bins-15))
  print("unique_array",unique.size, unique)
  print("new dist_array",dist_array[0:14])
  dist_array_kpc=dist_array*0.0159
  y_expect=(dist_array**(1.0/3))*34
  y_expect2=(dist_array**(1.0/2))*22
  y_expect3=(dist_array**(2.0/3))*14
  y_expect4=(dist_array*4)
  plt.clf()
  f = plt.figure(figsize = (10,8))
  plt.loglog(dist_array_kpc,y_expect,linestyle="-",label="1/3")
  plt.loglog(dist_array_kpc,y_expect4,label="1")
  plt.plot([0.052,0.104],[27,27],linestyle="-",color="C5",label="Simionescu19",marker="|",linewidth=2)
  v_diff_mean=np.zeros(n_bins)
  v_diff_mean2=np.zeros(n_bins)
  error_mean=np.zeros(n_bins)
  for i in range(0,n_bins-1):
    this_bin=(np_dist>=dist_array[i])&(np_dist<dist_array[i+1])
    v_diff_mean[i]=np.mean(np.abs(np_v_diff[this_bin]))
    v_diff_mean2[i]=np.mean((np_v_diff[this_bin])**2)
    error_mean[i]=np.sqrt(np.mean(np_error_2[this_bin]))
    print(dist_array_kpc[i],np_v_diff[this_bin].size,error_mean[i])
  error_mean_smooth=run_helper(dist_array_kpc,error_mean)
  v_diff_mean_smooth=run_helper(dist_array_kpc,v_diff_mean)
  lower_v=v_diff_mean_smooth-error_mean_smooth
  upper_v=v_diff_mean_smooth+error_mean_smooth
  plt.loglog(dist_array_kpc,v_diff_mean,marker="o",linestyle="None",markersize=4,color="C0")
  plt.fill_between(dist_array_kpc[:-19],lower_v[:-19],upper_v[:-19],color="C0",alpha=0.2)
  plt.xlabel("separation (kpc)")
  plt.ylabel(r"$|\delta v|\, \rm (km/s)$")
  plt.legend(loc="lower right", prop={'size': 16})
  plt.ylim(4,200)
  plt.xlim(dist_array_kpc[0]*0.9,6)
  plt.grid()
  plt.savefig("VSF_Virgo.png")
  plt.clf()
  fig, ax = plt.subplots(figsize = (8,8))
  ax.hist(np_dist*res,bins=200,range=(0,6))
  ax.set_xlabel("separation (kpc)")
  ax.set_ylabel("number of pairs")
  ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
  ax.set_title("Virgo",size=20)
  plt.tight_layout()
  plt.savefig("Virgo_separation_hist.png")

  if cut is True:
    np.savez("Virgo_cut",dist_array=dist_array,v_diff_mean=v_diff_mean)
  else:
    np.savez("Virgo_final",dist_array_kpc=dist_array_kpc,v_diff_mean=v_diff_mean,v_diff_mean2=v_diff_mean2,lower_v=lower_v,upper_v=upper_v,v_diff_mean_smooth=v_diff_mean_smooth)
  return

def nan_helper(y):
  return np.isnan(y), lambda z: z.nonzero()[0]
def run_helper(x,y):
  nans, x= nan_helper(y)
  z=np.copy(y)
  z[nans]= np.interp(x(nans), x(~nans), y[~nans])
  return(z)

make_vsf_plot(d_max=600,n_bins=200,check_Guassian=False)
#check_Guassian is a stub. I used to plot out all the distributions of delta_v to see if it is a Guassian. You can add your own calculations there to check Guassianity and other things related to the statistics of delta v at a given l. 

