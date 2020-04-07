### WaveSplit

things currently not clear:
---

- what is embedding dimension ? It seems 512 but it is not explicit in the paper
- mask used (sigmoid ?)
- when speakers in an example < sep stack outputs loss is simply masked or an embedding for silence is used ? (Probably masked)
- is VAD used in WSJ02MiX/ WHAM for determining speech activity at frame level ? Some files can have pauses of even one second
- loss right now is prone to go NaN especially if we don't take the mean after l2-distances computation. 
- FROM THE PAPER IT SEEMS BOTTLENECK CONVS ARE NOT USED AT ALL KEEPING IT FOR NOW