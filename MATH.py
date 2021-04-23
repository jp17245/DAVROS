def calc():
    groups =[[233, 124, 333, 329], [238, 104, 331, 306], [284, 141, 376, 359], [286, 123, 378, 336], [263, 138, 354, 367],[257, 116, 357, 341], [265, 124, 359, 376]]
    counter=-1
    smallestX=10000
    smallestY=10000
    largeX=0
    largeY=0

    smallArea=10000000
    largeArea=0
    for i in groups:
            print("i",i)
            group_centre_x = i[0] + (i[2] - i[0]) / 2
            group_centre_y = i[1] + (i[3] - i[1]) / 2
            area_old = (i[2] - i[0]) * (i[3] - i[1])

            if(group_centre_x >largeX):
                largeX =group_centre_x
            if(group_centre_x<smallestX):
                smallestX = group_centre_x

            if (group_centre_y > largeY):
                largeY = group_centre_y
            if (group_centre_y < smallestY):
                smallestY = group_centre_y

            if (area_old > largeArea):
                largeArea = area_old
            if (area_old < smallArea):
                smallArea = area_old
            print("Centre x: ", group_centre_x)
            print("Centre y: ", group_centre_y)
            print("Area", area_old)
            print("---------------------------------------------------------------------------------------")
    print("Smallest X : ", smallestX)
    print("Smallest Y : ", smallestY)
    print("Largest X : ", largeX)
    print("Largest Y : ", largeY)
    print("Largest Area: ", largeArea)
    print("Smallest Area : ", smallArea)

calc()