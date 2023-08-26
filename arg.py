import argparse
import sys
sys.argv=['']

parser = argparse.ArgumentParser(description="Semantic segmentation")
parser.add_argument("-r","--root",type = str, default = "/home/hardik/data")
# parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
# parser.add_argument("-b","--batch_num_per_class",type = int, default = 5)
# parser.add_argument("-e","--episode",type = int, default= 500000)
# parser.add_argument("-t","--test_episode", type = int, default = 1000)
# parser.add_argument("-l","--learning_rate", type = float, default = 0.0001)
# parser.add_argument("-rf","--TrainResultPath",type=str,default='result_5shot')
# parser.add_argument("-rff","--ResultSaveFreq",type=int,default=10000)
# parser.add_argument("-msp","--ModelSavePath",type=str,default='models_5shot')
# parser.add_argument("-msf","--ModelSaveFreq",type=int,default=10000)
# parser.add_argument("-g","--gpu",type=int, default=0)
# parser.add_argument("-d","--display_query_num",type=int,default=5)
# parser.add_argument("-modelf","--encoder_model",type=str,default='B1_model/efficientnet_encoder_99999_1_way_5shot.pkl')
# parser.add_argument("-modeld","--decoder_model",type=str,default='B1_model/decoder_network_99999_1_way_5shot.pkl')
# parser.add_argument("-modelh","--head_model",type=str,default='B1_model/segmentation_network_99999_1_way_5shot.pkl')
# parser.add_argument("-modelc","--critic_model",type=str,default='')
# parser.add_argument("-start","--start_episode",type = int, default= 0)
# parser.add_argument("-fi","--finetune",type=bool,default=True)


args = parser.parse_args()

