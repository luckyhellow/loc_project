#include "tools.h"

Qlabel_List::Qlabel_List()
{

}

bool insert_intail(QlabelList& head,QLabel* label)//
{
    if(head == NULL)
    {
        head = new Node;
        head->label = label;
        head->next = NULL;
        return 1;
        return 0;
    }
    else
    {   Node *q;
        Node *p;
        for(p=head;p!=NULL;p=p->next)
        {
            q=p;
        }
        q->next = new Node;
        q->next->next = NULL;
        q->next->label = label;
        return 1;
    }
    return 0;
}

void clear(QlabelList& head)
{
    if(head==NULL) return;
    Node* p = head;
    Node* q = NULL;
    while(p!=NULL)
    {
        q = p;
        p = p->next;
        q->label->clear();
        //deal with Memory leak
        delete q->label;
        delete q;
    }
    head = NULL;
}
