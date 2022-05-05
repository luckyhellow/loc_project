#include "time_update.h"

Time_update::Time_update(QObject *parent) : QObject(parent)
{

}
//void Time_update:: begin_update()
//{
//    timeupdate->start(500);
//}
void Time_update:: begin_update(int hz){
    timeupdate1->start(1000/hz);
}
void Time_update:: begin_recv(int hz){
    timeupdate2->start(1000/hz);
}
