# VStego800K

VStego800K: Large-scale Steganalysis Dataset for Streaming Voice,mixed with various steganographic algorithms, embedding rates, and quality factors.   
In recent years, more and more steganographic methods based on streaming voice have appeared, which poses a great threat to the security of cyberspace. In this paper, in order to promote the development of streaming voice steganalysis technology, we construct and release a large-scale streaming voice steganalysis dataset called VStego800K. To truly reflect the needs of reality, we mainly follow three considerations when constructing the VStego800K dataset: large-scale, real-time, and diversity. The large-scale dataset allows researchers to fully explore the statistical distribution differences of streaming signals caused by steganography. Therefore, the proposed VStego800K dataset contains 814,592 streaming voice fragments. Among them, 764,592 samples (382,296 cover-stego pairs) are divided as the training set and the remaining 50,000 as testing set. The duration of all samples in the data set is uniformly cut to 1 second to encourage researchers to develop near real-time speech steganalysis algorithms. To ensure the diversity of the dataset, the collected voice signals are mixed with male and female as well as Chinese and English from different speakers. For each steganographic sample in VStego800K, we randomly use two typical streaming voice steganography algorithms, and randomly embed random bit with embedding rates of 10%-40%. We tested the performance of some latest steganalysis algorithms on VStego800K, with specific results and analysis details in the experimental part. We hope that the VStego800K dataset will further promote the development of universal voice steganalysis technology.


## Download

####Google Drive Link: [Train Set](https://drive.google.com/drive/folders/1IhpCFH0e5IkMzpm48IzVBDAN2VVnjSKI?usp=sharing) 

382,296 pairs of cover and stego   
CNV embedding rates at 10%:45875 samples   
CNV embedding rates at 20%:49698 samples   
CNV embedding rates at 30%:45877 samples    
CNV embedding rates at 40%:49700 samples  
PMS embedding rates at 10%:45877 samples  
PMS embedding rates at 20%:49696 samples    
PMS embedding rates at 30%:45876 samples    
PMS embedding rates at 40%:49700 samples 

####Google Drive Link: [Test Set](https://drive.google.com/drive/folders/1RD7yOHtCgmb8BgP4EDTT3v1d3mV48mmu?usp=sharing) 

25000 cover   
25000 stego 


#### Alternate links

For those who cannot access Google in Mainland China, try this Baidu Cloud Disk link:  

*Baidu Drive Link: [Train Set](https://pan.baidu.com/s/1dJtBXQuZnG2eba13tbmnOA)(__extraction code__:a1xd)   
*Baidu Drive Link: [ Test Set](https://pan.baidu.com/s/1MREl-doUf2MG4-BuE91P0w)(__extraction code__:levg

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

  

  __Overall Results__

   The overall performance of each steganalysis methods on test set.

  | **Steganalysis Method** | Accuracy | Precision | Recall |   F1   |
  | :---------------------: | :------: | --------- | ------ | :----: |
  |         SS-QCCN         |  0.6117  | 0.6595    | 0.4617 | 0.5432 |
  |           CCN           |  0.5542  | 0.5544    | 0.5517 | 0.5531 |
  |           RSM           |  0.5174  | 0.5103    | 0.8605 | 0.6407 |
  |           FSM           |  0.7094  | 0.7085    | 0.7115 | 0.7100 |
  |          SFFN           |  0.7048  | 0.7206    | 0.6689 | 0.6938 |

  The steganalysis results of benchmark methods for different embedding rates in VStego800K .

  | Steganalysis  Embedding  Rates |      | 10%    | 20%    | 30%    | 40%    |
  | ------------------------------ | ---- | ------ | ------ | ------ | ------ |
  | SS-QCCN                        | Acc  | 0.6792 | 0.6933 | 0.7189 | 0.7149 |
  |                                | R    | 0.3274 | 0.4224 | 0.5279 | 0.5533 |
  | CCN                            | Acc  | 0.5510 | 0.5436 | 0.5582 | 0.5495 |
  |                                | R    | 0.5267 | 0.4917 | 0.5651 | 0.6138 |
  | RSM                            | Acc  | 0.3029 | 0.3157 | 0.3087 | 0.3186 |
  |                                | R    | 0.8390 | 0.8595 | 0.8687 | 0.8737 |
  | FSM                            | Acc  | 0.6683 | 0.7026 | 0.7232 | 0.7381 |
  |                                | R    | 0.5057 | 0.6847 | 0.7893 | 0.8568 |
  | SFFN                           | Acc  | 0.6865 | 0.7171 | 0.7411 | 0.7601 |
  |                                | R    | 0.4610 | 0.6265 | 0.7340 | 0.8349 |




