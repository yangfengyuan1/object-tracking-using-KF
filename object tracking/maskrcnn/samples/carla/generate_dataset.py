import random, csv

l=list(range(7500))
l1=random.sample(l,len(l)*6//10)
l1.sort()
l=list(set(l).difference(set(l1)))
l2=random.sample(l,len(l)*1//2)
l2.sort()
l3=list(set(l).difference(set(l2)))

l=list(range(3918))
l4=random.sample(l,len(l)*6//10)
l4.sort()
l=list(set(l).difference(set(l4)))
l5=random.sample(l,len(l)*1//2)
l5.sort()
l6=list(set(l).difference(set(l5)))

with open('train.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['prefix','id'])
    for x in l1:
        if x%3==0:
            writer.writerow(['ClearDay/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearDay/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearDay/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])

    for x in l4:
        if x%3==0:
            writer.writerow(['ClearNight/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearNight/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearNight/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])

with open('val.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['prefix','id'])
    for x in l2:
        if x%3==0:
            writer.writerow(['ClearDay/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearDay/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearDay/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])

    for x in l5:
        if x%3==0:
            writer.writerow(['ClearNight/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearNight/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearNight/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])

with open('test.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(['prefix','id'])
    for x in l3:
        if x%3==0:
            writer.writerow(['ClearDay/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearDay/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearDay/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])

    for x in l6:
        if x%3==0:
            writer.writerow(['ClearNight/Town10HD_Opt_83.3_-63_7_180.0_-15.0_0.0/','%06d'%x])
        elif x%3==1:
            writer.writerow(['ClearNight/Town10HD_Opt_-92.91_19.21_7_0.0_-15.0_0.0/','%06d'%x])
        else:
            writer.writerow(['ClearNight/Town10HD_Opt_-5.16_133.67_7_180.0_-15.0_0.0/','%06d'%x])
