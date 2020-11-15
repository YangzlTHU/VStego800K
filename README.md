# VStego800K

VStego800K: Large-scale Steganalysis Dataset for Streaming Voice,mixed with various steganographic algorithms, embedding rates, and quality factors.   
In recent years, more and more steganographic methods based on streaming voice have appeared, which poses a great threat to the security of cyberspace. In this paper, in order to promote the development of streaming voice steganalysis technology, we construct and release a large-scale streaming voice steganalysis dataset called VStego800K. To truly reflect the needs of reality, we mainly follow three considerations when constructing the VStego800K dataset: large-scale, real-time, and diversity. The large-scale dataset allows researchers to fully explore the statistical distribution differences of streaming signals caused by steganography. Therefore, the proposed VStego800K dataset contains 814,592 streaming voice fragments. Among them, 764,592 samples (382,296 cover-stego pairs) are divided as the training set and the remaining 50,000 as testing set. The duration of all samples in the data set is uniformly cut to 1 second to encourage researchers to develop near real-time speech steganalysis algorithms. To ensure the diversity of the dataset, the collected voice signals are mixed with male and female as well as Chinese and English from different speakers. For each steganographic sample in VStego800K, we randomly use two typical streaming voice steganography algorithms, and randomly embed random bit with embedding rates of 10%-40%. We tested the performance of some latest steganalysis algorithms on VStego800K, with specific results and analysis details in the experimental part. We hope that the VStego800K dataset will further promote the development of universal voice steganalysis technology.


## Download

#### [Train Set](https://drive.google.com/drive/folders/1IhpCFH0e5IkMzpm48IzVBDAN2VVnjSKI?usp=sharing) 
382,296 pairs of cover and stego   
CNV embedding rates at 10%:45875 samples   
CNV embedding rates at 20%:49698 samples   
CNV embedding rates at 30%:45877 samples    
CNV embedding rates at 40%:49700 samples  
PMS embedding rates at 10%:45877 samples  
PMS embedding rates at 20%:49696 samples    
PMS embedding rates at 30%:45876 samples    
PMS embedding rates at 40%:49700 samples 
#### [Test Set](https://drive.google.com/drive/folders/1RD7yOHtCgmb8BgP4EDTT3v1d3mV48mmu?usp=sharing) 
25000 cover   
25000 stego 


#### Alternate links
For those who cannot access Google in Mainland China, try this Baidu Cloud Disk link:  
* [Train Set](https://pan.baidu.com/s/1dJtBXQuZnG2eba13tbmnOA)(__extraction code__:a1xd)   
* [ Test Set](https://pan.baidu.com/s/1MREl-doUf2MG4-BuE91P0w)(__extraction code__:levg

#### Detailed Parameters
We also provide detailed parameters for each samples in corresponding label.csv
    
## Steganographic Algorithms 

We use the following steganographic algorithms for our dataset:
 
* __CNV-QIM__:Xiao B, Huang Y, Tang S. An approach to information hiding in low bit-rate speech stream[C]//IEEE GLOBECOM 2008-2008 IEEE Global Telecommunications Conference. IEEE, 2008: 1-5.
- __PMS__: Huang Y, Liu C, Tang S, et al. Steganography integration into a low-bit rate speech codec[J]. IEEE transactions on information forensics and security, 2012, 7(6): 1865-1875.

For more details, including codes and tutorial, please refer to our __[Steganography page](Steganography)__.

## Steganalysis Algorithms
We apply the following steganalysis algorithms for dataset evaluation: 

* __SS-QCCN__:Yang, H., Yang, Z., Bao, Y., & Huang, Y. (2019, December). Hierarchical representation network for steganalysis of qim steganography in low-bit-rate speech signals. In International Conference on Information and Communications Security (pp. 783-798). Springer, Cham.
- __CCN__:Li, S. B., Jia, Y. Z., Fu, J. Y., & Dai, Q. X. (2014). Detection of pitch modulation information hiding based on codebook correlation network. Chinese Journal of Computers, 37(10), 2107-2117.
* __RSM__:Lin, Z., Huang, Y., & Wang, J. (2018). RNN-SM: Fast steganalysis of VoIP streams using recurrent neural network. IEEE Transactions on Information Forensics and Security, 13(7), 1854-1868. 
- __FSM__:Yang, H., Yang, Z., Bao, Y., Liu, S., & Huang, Y. (2019). Fast steganalysis method for voip streams. IEEE Signal Processing Letters, 27, 286-290.
- __SFFN__:Hu, Y., Huang, Y., Yang, Z., & Huang, Y. Detection of heterogeneous parallel steganography for low bit-rate VoIP speech streams. Neurocomputing, 419, 70-79.
For more details, including codes and tutorial, please refer to our __[Steganalysis page](Steganalysis)__.


