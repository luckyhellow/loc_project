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
    //fluence of updating xy's value
//    TU->begin_recv(1000);

    //输入框
    QLineEdit* lineedit = new QLineEdit();
    QIntValidator *intValidator = new QIntValidator;
    //setting the legal range
    intValidator->setRange(1, 1000);
    lineedit->setValidator(intValidator);
    lineedit->setParent(this);
    lineedit->resize(100,27);
    lineedit->move(100,670);
    lineedit->show();

    //creat a new instance of "udprecv" to recv the data
    udp = new UDPrecv();

    QMovie* movie = new QMovie(":/Image/point.gif");
    movie->start();
    connect(TU->timeupdate, &QTimer::timeout,[=](){//fluent to show
        //get the location of xy now
        location_xy = udp->getxy();
        //creat a label and use it show the xy's location
        //watch out Memory leak!
        label = new QLabel(this);
        label->setMovie(movie);
        label->setAlignment(Qt::AlignCenter);
        label->resize(10,10);
        label->move(140+location_xy[0],location_xy[1]+40);
        label->show();
        //add to list inorder to realize the function of deleting labels
        insert_intail(Listhead,label);
    });

    //create button to control start or not
    //and connect it with HZ of showing label
    MyPushButton *startButton = new MyPushButton(":/Image/startButton.jpg");
    startButton->setParent(this);
    startButton->move(300,625);
    connect(startButton,&MyPushButton::clicked,[=](){
        //a simple animation effect
        startButton->tik();
        startButton->tok();
        if(lineedit->text()==""){
            TU->begin_update();
//            qDebug() <<"str == null\n";
        }
        else {
            hz = lineedit->text().toInt();
//            qDebug() <<hz;
            TU->begin_update(hz);
        }
    });

    //create button to clear the labels we have created
    MyPushButton *clearButton = new MyPushButton(":/Image/clearButton.jpg");
    clearButton->setParent(this);
    clearButton->move(625,625);
    connect(clearButton,&MyPushButton::clicked,[=](){
        //a simple animation effect
        clearButton->tik();
        clearButton->tok();
        //clear the show and delete labels we create with "new"
        //watch out memory leak
        clear(Listhead);
    });

}

void MainWindow::paintEvent(QPaintEvent *)
{
    //paint the backgroud
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

