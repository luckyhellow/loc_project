#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
//    Py_SetPythonHome(L"C:/ProgramData/Anaconda3/");
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
