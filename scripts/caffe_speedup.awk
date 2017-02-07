#!/usr/bin/awk -f
/x86_64_caffe_time/{shift=NR-1}
/armv7l_caffe_time/{shift=NR-1}
{
    if (NR==3+shift) ct1=$3
    if (NR==4+shift) ct2=$3

    if (NR==8+shift) $0=$0" (x"ct1/$3")"
    if (NR==9+shift) $0=$0" (x"ct2/$3")"

    if (NR==13+shift) ca1=$3
    if (NR==14+shift) ca2=$3
    if (NR==15+shift) ca3=$3
    if (NR==16+shift) ca4=$3
    if (NR==17+shift) ca5=$3

    if (NR==21+shift) $0=$0" (x"ca1/$3")"
    if (NR==22+shift) $0=$0" (x"ca2/$3")"
    if (NR==23+shift) $0=$0" (x"ca3/$3")"
    if (NR==24+shift) $0=$0" (x"ca4/$3")"
    if (NR==25+shift) $0=$0" (x"ca5/$3")"

    print
}
