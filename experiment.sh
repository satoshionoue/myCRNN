#!/bin/bash

# 入力音源ファイルを引数で受け取る
#使い方：bash experiment.sh in_dir/audiotechnica.wav

if [ $# -ne 1 ]; then
  echo "使い方: $0 入力音源ファイル"
  exit 1
fi

INPUT_WAV="$1"
OUT_DIR="out_dir1/5_actual_epoch1"

# 出力ディレクトリがなければ作成
if [ ! -d "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
fi

# 1. 音源をメルスペクトログラムに変換
python Acoustic/data/wav_to_mel.py "$INPUT_WAV" "$OUT_DIR/mel.npy" 10

# 2. メルスペクトログラムの連結
python Acoustic/data/concat_estimate.py "$OUT_DIR/mel.npy" "$OUT_DIR/mel_conc.npy"

# 3. ピッチ・オンセットそれぞれの推論
python Acoustic/CRNN/my_crnn_estimate.py Acoustic/CRNN/config_CRNN_ex_onset.yaml res_CRNN_ex/o/checkpt_1.pt "$OUT_DIR/mel_conc.npy" "$OUT_DIR/mel_o.txt"
python Acoustic/CRNN/my_crnn_estimate.py Acoustic/CRNN/config_CRNN_ex_pitch.yaml res_CRNN_ex/p/checkpt_1.pt "$OUT_DIR/mel_conc.npy" "$OUT_DIR/mel_p.txt"

# 4. ピアノロールの生成
python Acoustic/simple_note_tracker.py "$OUT_DIR/mel_p.txt" "$OUT_DIR/mel_o.txt" "$OUT_DIR/ipr.txt"

# 5. IPRからMIDIデータの生成
./Acoustic/MIDI_Pianoroll/pianoroll2midi "$OUT_DIR/ipr.txt" "$OUT_DIR/est.mid"