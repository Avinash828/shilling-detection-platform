# coding=utf-8
from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from RR_Attack import RR_Attack
from hybridAttack import HybridAttack

attack = RandomAttack('./config/config.conf') # 随机攻击
# attack = AverageAttack('./config/config.conf') # 平均攻击
# attack = BandWagonAttack('./config/config.conf') # 流行攻击

attack.insertSpam()
# attack.farmLink()
attack.generateLabels('labels.txt') # 标签数据 具体格式 userid rating
attack.generateProfiles('profiles.txt') # 评分文件 具体格式 userid itemid rating

# attack.generateSocialConnections('relations.txt')
