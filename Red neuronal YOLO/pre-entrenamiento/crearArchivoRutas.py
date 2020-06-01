import os 


folder='/home/pc/Escritorio/darknet 9010/img/test'
archivos = []
ruta='img/test/'

archivos=os.listdir(folder)
name="test.txt"

file = open(name, "w")

for i in archivos:
	tam = len(i)
	formato = i[tam-4:tam]
	if(formato == ".jpg"):
		file.write(ruta+i +'\n')

file.close()
