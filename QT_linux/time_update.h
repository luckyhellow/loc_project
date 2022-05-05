#ifndef TIME_UPDATE_H
#define TIME_UPDATE_H

#include <QTimer>
#include <QObject>

class Time_update : public QObject
{
    Q_OBJECT
public:
    QTimer *timeupdate1 = new QTimer(this);
    QTimer *timeupdate2 = new QTimer(this);
    explicit Time_update(QObject *parent = nullptr);
//    void begin_update();
    void begin_update(int hz = 1);
    void begin_recv(int hz = 1000);
};

#endif // TIME_UPDATE_H
