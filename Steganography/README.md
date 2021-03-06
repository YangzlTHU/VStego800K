
# VoIP-Stega

This tool can be used for VoIP speech steganography. It contains two steganography algorithms, CNV-QIM [1] and Pitch modulation [2].
At the same time, this tool is also used to build the first National Information Hiding Competition (CIHC2019) audio training set.

The usage is as follows:

1. Use wav2pcm.py to remove the wav header and change the wav file into a pcm file. You need to modify the path of wav file  and the save path of pcm in. py file.

2. StegCoder. Exe is used for steganography

   (1) Place the pcm file that requires steganography in a folder named Input.

   (2)  Create a new output folder for steganized files.

   (3)  If you are using CNV-QIM steganography algorithms with a 100% 	embedding rate, type at the command prompt:

   ​       `StegCoder.exe -i input -o output -a cnv -r 100 `

   ​	 If you are using Pitch modulation steganography algorithms with a 100% embedding rate, type at the command prompt:

   ​     ` StegCoder.exe -i input -o output -a pitch -r 100`

   

   If you use the Steganography tool in your academic research, please quote the following paper:

   【1】Xiao B, Huang Y, Tang S. An approach to information hiding in low bit-rate speech stream[C]//IEEE GLOBECOM 2008-2008 IEEE Global Telecommunications Conference. IEEE, 2008: 1-5.

   【2】Huang Y, Liu C, Tang S, et al. Steganography integration into a low-bit rate speech codec[J]. IEEE transactions on information forensics and security, 2012, 7(6): 1865-1875.

使用方法如下：

1. 使用wav2pcm.py去掉wav文件头，将wav文件变成pcm文件，需要修改.py文件中wav文件的路径和pcm的保存路径。
2. 使用StegCoder.exe进行隐写
   （1）将需要隐写的pcm文件放在一个命名为input的文件夹内  
   （2）新建一个output文件夹，用于存放隐写后的文件  
   （3）如果使用CNV-QIM隐写，隐写嵌入率为100%，则在命令提示符内输入 
   StegCoder.exe -i input -o output -a cnv -r 100  
   如果使用基音隐写方法，隐写嵌入率为100%，则在命令提示符内输入   
   StegCoder.exe -i input -o output -a pitch -r 100  

注1：wav文件为8000HZ采样，16bit量化。  
注2：输入和输出文件夹的名字可以自己指定，注意两种隐写数据不要指定在同一个输出文件夹下，可能会出现相同文件名而互相覆盖的问题。   


若您在学术研究中用到该隐写工具，请引用如下论文：

【1】Xiao B, Huang Y, Tang S. An approach to information hiding in low bit-rate speech stream[C]//IEEE GLOBECOM 2008-2008 IEEE Global Telecommunications Conference. IEEE, 2008: 1-5.

【2】Huang Y, Liu C, Tang S, et al. Steganography integration into a low-bit rate speech codec[J]. IEEE transactions on information forensics and security, 2012, 7(6): 1865-1875.

===========================================================================================

