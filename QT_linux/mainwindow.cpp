#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "mypushbutton.h"
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/Image/Icon.jpg"));
    //重置窗口大小
    resize(1080,800);
    //固定大小
    setFixedSize(1080,800);
    //设置窗口标题
    setWindowTitle("(lucky_demo)locate app");


      // //////////
     //   菜单   //   start
    // //////////


    //创建菜单栏
    QMenuBar* bar = menuBar();
    //菜单栏放入窗口
    setMenuBar(bar);
    //创建菜单
    QMenu *fileMenu = bar->addMenu("功能");
    QMenu *editMenu = bar->addMenu("编辑");

    fileMenu->addAction("尚未完善");
    fileMenu->addSeparator();
    fileMenu->addAction("尚未完善");
    editMenu->addAction("尚未完善");

      // //////////
     //   菜单   //   end
    // //////////

    this->TU = new Time_update(this);//计时器更新的频率

    //输入框
    QLineEdit* lineedit = new QLineEdit();
    QIntValidator *intValidator = new QIntValidator;
    intValidator->setRange(1, 10000);
    lineedit->setValidator(intValidator);
//    lineedit->setValidator( new QIntValidator(0, 10000, this) );
    lineedit->setParent(this);
    lineedit->resize(100,27);
    lineedit->move(100,670);
    lineedit->show();
    //实例化calculate类
//    Calculate* calculate = new Calculate();
    udp = new UDPrecv();
    connect(TU->timeupdate, &QTimer::timeout,[=](){
//        qDebug()<<"1111";
        location_x = udp->getx();
        location_y = udp->gety();
        //后续改为读取传进来的参数
        label = new QLabel(this);
        QMovie* movie = new QMovie(":/Image/point.gif");
        label->setMovie(movie);
        label->setAlignment(Qt::AlignCenter);
        label->resize(10,10);
        label->move(140+location_x,location_y+40);
        movie->start();
        label->show();
        insert_intail(Listhead,label);
    });

    MyPushButton *startButton = new MyPushButton(":/Image/startButton.jpg");
    startButton->setParent(this);
    startButton->move(300,625);
    connect(startButton,&MyPushButton::clicked,[=](){
//        qDebug() << "呵呵";
        startButton->tik();
        startButton->tok();
        if(lineedit->text()==""){
            TU->begin_update();
            qDebug() <<"str == null\n";
        }
        else {
            hz = lineedit->text().toInt();
            qDebug() <<hz;
            TU->begin_update(hz);
        }
    });

    MyPushButton *clearButton = new MyPushButton(":/Image/clearButton.jpg");
    clearButton->setParent(this);
    clearButton->move(625,625);
    connect(clearButton,&MyPushButton::clicked,[=](){
//        qDebug() << "呵呵";
        clearButton->tik();
        clearButton->tok();
        clear(Listhead);
    });

}

void MainWindow::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    QPixmap pix;
    pix.load(":/Image/floor.jpg");
    painter.drawPixmap(0,0,this->width(),this->height(),pix);

    pix.load(":/Image/white.jpg");
    pix = pix.scaled(800,600);
    painter.drawPixmap(140,0,800,600,pix);

    pix.load(":/Image/background.jpg");
    pix = pix.scaled(1080,200);
    painter.drawPixmap(0,600,1080,200,pix);

}


MainWindow::~MainWindow()
{
    delete ui;
}

