#include "tools.h"

void insert(QlabelList& head,QLabel* qlabel)//
{
    if(head == NULL)
    {
        head = new Node;
        head->qlabel = qlabel;
        head->next = NULL;
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
        q->next->qlabel = qlabel;
    }
    return;
}

void clean(QlabelList& head)
{
    if(head==NULL) return;
    Node* p = head;
    Node* q = NULL;
    while(p!=NULL)
    {
        q = p;
        p = p->next;
        q->qlabel->clear();
        //deal with Memory leak
        delete q->qlabel;
        delete q;
    }
    head = NULL;
}

void hideqlabels(QlabelList& head){
    if(head==NULL) return;
    Node* p = head;
    while(p!=NULL)
    {
        p->qlabel->hide();
        p = p->next;
    }
}

void showqlabels(QlabelList& head){
    if(head==NULL) return;
    Node* p = head;
    while(p!=NULL)
    {
        p->qlabel->show();
        p = p->next;
    }
}
