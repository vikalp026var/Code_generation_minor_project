
import io
from src.transformer.Encoder1 import Encoder
from src.transformer.Decoder1 import Decoder
from src.transformer.Seq2seq import Seq2Seq
from dataclasses import dataclass

@dataclass
class Model:
     def model(self):
          INPUT_DIM = 2114
          OUTPUT_DIM = 5670
          HID_DIM = 256
          ENC_LAYERS = 3
          DEC_LAYERS = 3
          ENC_HEADS = 16
          DEC_HEADS = 16
          ENC_PF_DIM = 512
          DEC_PF_DIM = 512
          ENC_DROPOUT = 0.1
          DEC_DROPOUT = 0.1
          device="cpu"


          enc = Encoder(INPUT_DIM,
                    HID_DIM,
                    ENC_LAYERS,
                    ENC_HEADS,
                    ENC_PF_DIM,
                    ENC_DROPOUT,
                    device)

          dec = Decoder(OUTPUT_DIM,
                    HID_DIM,
                    DEC_LAYERS,
                    DEC_HEADS,
                    DEC_PF_DIM,
                    DEC_DROPOUT,
                    device)

          model = Seq2Seq(enc, dec, 1, 1, device).to(device)
          return model


