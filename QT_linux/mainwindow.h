#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPainter>
#include <QTimer>
#include <QLabel>
#include <QMovie>
#include "time_update.h"
#include "qlabel_list.h"
#include <QIntValidator>
#include <QLineEdit>
#include <udprecv.h>
#include <vector>

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
    Ui::MainWindow *ui;
    Time_update *TU;
    vector<double> location_xy = {0,0};
    QLabel *label;
    labelList Listhead = NULL;
    int hz = 0;
    UDPrecv* udp;
    int wait = 0;
};
#endif // MAINWINDOW_H
