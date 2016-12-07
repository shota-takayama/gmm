COMPILER = g++
INCLUDE  = -I/usr/local/Cellar/opencv/2.4.12/include/opencv2
LIBS = -L/usr/local/Cellar/opencv/2.4.12/lib -lopencv_core -lopencv_highgui
SOURCES = main.cpp gmm.cpp
TARGET = gmm

$(TARGET): $(SOURCES)
	$(COMPILER) $(INCLUDE) $(LIBS) $(SOURCES) -o $(TARGET)
