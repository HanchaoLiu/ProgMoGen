name_list=('ACCAD.tar.bz2'  'BMLmovi.tar.bz2'  'CMU.tar.bz2'    'EKUT.tar.bz2'  'HDM05.tar.bz2'    'KIT.tar.bz2' 'MoSh.tar.bz2'      'SFU.tar.bz2'  'TCDHands.tar.bz2'      'Transitions.tar.bz2' 'BMLhandball.tar.bz2'  'BMLrub.tar.bz2'   'DFaust.tar.bz2' 'EyesJapanDataset.tar.bz2'  'HumanEva.tar.bz2'  'PosePrior.tar.bz2'  'SSM.tar.bz2'  'TotalCapture.tar.bz2')

for name in ${name_list[@]}; do

echo "tar -xvf ${name}"

done