#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QTimer>
#include <QLabel>
//#include <QMovie>
#include "time_update.h"
#include "tools.h"
#include <QIntValidator>
#include <QLineEdit>
#include <udprecv.h>
#include <vector>
#include <unordered_map>
#include <QGroupBox>
#include <QCheckBox>
#include <QVBoxLayout>
#include "mypushbutton.h"
#include <QImage>
#include <random>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void paintEvent(QPaintEvent *);

private:
    QCheckBox *checkbox;
    double scale;
    Ui::MainWindow *ui;
    Time_update *TU;
    Struct_XY struct_xy = {};
    QLabel *qlabel;
    int hz = 0;
    UDPrecv* udp;
    int wait = 0;
    unordered_map<string,QlabelList> hashmap;
    unordered_map<string,bool> hashchoose;
    unordered_map<string,QPixmap> hashpix;
    vector<QCheckBox *> qcheckboxs;
};
#endif // MAINWINDOW_H
