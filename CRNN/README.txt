[CRNN]


[description]

Pytorch CRNN (RNN = bidirectional LSTM)

input スペクトログラム 複数枚(n_chan) -> multilabel の 分類

Input: x[n][c][l][i] (float32)
n: sequence index n = 1,...,N
c: chan index c = 1,...,n_chan
l: location idx l=1,...,L (学習のために系列長は揃えておく。zero paddingなど適用。)
i: feature idx i=1,...,I (I=in_dim)

Target: y[n][l][k] (int)
n,l はinputと対応
k: label class idx k=1,...,K
y[n,l,k] \in Omega_k = {1,...,N_k} (class k の vocabulary)
Nk: label set size

Output of CRNN: a[n,l,k,j] (float)
activation vectors = predictive probabilities
n, l, k はtargetと対応
j=1,...,N_k はclass cの要素のidx
a[n,l,k,j]はprobability vector \sum_{j=1}^{N_k} a[n,l,k,j] = 1
Pytorchのforwardの出力ではsoftmax前の activation a~[n,l,k,j]にすることが多い。

Input dataとtarget dataはnpyファイルで用意。
Input data: N x C x L x I (float32)
Target data: N x L x K (int)
Nは系列の本数。

[data preparation (random data)]

python setup_npy_data_random.py ex_train_input.npy ex_train_target.npy
python setup_npy_data_random.py ex_valid_input.npy ex_valid_target.npy
-> input [15, 2, 10, 5] target [15, 10, 1(12)]

[学習]

python my_crnn_train.py config_CRNN_ex.yaml 30

configに条件を書いておく。
最後の引数はエポック数。

lossは標準出力される。また、configで指定したフォルダー内にlossのlogファイルとCRNNのparameterが保存される。

[実行するプログラムの順番]
1. wav_to_input.py
- 音源をメルスペクトログラムに変換

2. ipr_to_lab_pitch_onset_list.py
- ipr.txtデータからターゲットのラベルとピッチオンセットのリストを作成

3. data_split.py
- メルスペクトログラムを分割して、npyファイルに保存

4. concat_CRNN.py
- configファイルを編集（ピッチとオンセットそれぞれについて実行する）
- 分割したnpyファイルを結合して、学習用のnpyファイルを作成

[推論の手順]
1. Acoustic/data/wav_to_mel.py
- 音源をメルスペクトログラムに変換
python 1_get_mel.py in.wav out_mel.npy 10

2. Acoustic/data/concat_estimate.py
-  
python concat_estimate.py in_mix in_vocal out_file

3. Acoustic/CRNN/my_crnn_estimate.py
- ピッチ・オンセットそれぞれの推論
python my_estimate.py config.yaml param.pt input.npy out_dir
python my_estimate.py config.yaml param.pt input.npy out_dir
