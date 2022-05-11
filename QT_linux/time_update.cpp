#include "time_update.h"

Time_update::Time_update(QObject *parent) : QObject(parent)
{

}
void Time_update:: begin_update(int hz){
    timeupdate->start(1000/hz);
}
