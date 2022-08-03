# WHU Stereo Dataset

## File Formats

The Stereo dataset project folder contains train and test set.
The train and test folder contain the following: 
```
train / test
.                          
├── 002_35              
│   ├── Left
│   │     ├── 000000.png
│   │     ├── 000001.png   
│   │     └── ...  
│   ├── Right      
│   │     ├── 000000.png
│   │     ├── 000001.png   
│   │     └── ...  
│   └── Disparity   
│          ├── 000000.png
│          ├── 000001.png   
│          └── ...               
├── 007_20              
│   ├── Left
│   │     ├── 000000.png
│   │     ├── 000001.png   
│   │     └── ...  
│   ├── Right      
│   │     ├── 000000.png
│   │     ├── 000001.png   
│   │     └── ...  
│   └── Disparity   
│          ├── 000000.png
│          ├── 000001.png   
│          └── ...  
└── ...           
.        
```
The first level folder took the name of the left image in a stereo unit, and the second folder were named as Left/Right/Disparity to store the sub-blocks and disparities.

### Image Files
Left and right sub-blocks are stored in the 'Left' and 'Right' folder respectively. 
We index each left block using a 6 digit number starting from `000000`.  The first three digits represent the Row index in the original image, and the last three digits represent the Col index in the original image. 
The correspinding right block and disparity block files use the same indexes as left block. 
The adjacent sub-blocks have 50% overlap in the X direction while 0% overlap in the Y direction.
According to the order of cropping, you can recover the original image from blocks.

### Disparity Files
The ground truth disparity map (left reference) of a pair of sub-block is stored in a ``xxxxxx.png`` file with 16 bit, using the same index as left block file. 
We stretched the disparity range by 256 times in order to maintain accuracy. 
The disparity value stored in 16 bit png file is 'STORED_DISPARITY_VALUE', while the ground truth disparity is 'TRUE_DISPARITY_VALUE', thus:

TRUE_DISPARITY_VALUE = (STORED_DISPARITY_VALUE / 256.0) 


