import vaex as vx
import pandas as pd
import numpy as np
import os
import pickle


def remove_duplicates(df, grouping_cols: list):
        """Removes duplicated rows based on groupping columns and created index.

        1. Create index.
        2. Create new data frame, group on grouping columns and find minimum index
        3. Join on index and index_min
        4. Remove rows without index_min (duplicated)
        5. Returns deduplicated data frame
        """
        df["index"] = vx.vrange(0, df.shape[0])
        df_group = df.groupby(grouping_cols, agg=vx.agg.min("index"))
        df = df.join(df_group[["index_min"]], left_on="index", right_on="index_min")
        df = df[df.index_min.notna()]

        df = df.drop(["index", "index_min"])
        df = df.extract()
        
        return df

def sort_pats_and_embs(file1:str,file3:str):
        ###opening grantreduced and embeddings
        df1 = vx.open(file1)
        
        ### sorting and creating new embbeding position as emb_pos
        df1=df1.sort(by='pub_date_days', ascending=True)
        df1["new_emb_pos"] = vx.vrange(0, df1.shape[0],dtype =int)

        ###sorting  embeddings
        embedding_pos=np.array(df1['embedding_pos'].values.tolist())

        #opening embeddings pickle file
        pkl_file = open(file3,'rb')
        test=pickle.load(pkl_file)
        pkl_file.close()

        #resorting embeddings from embedding_pos
        embeddings=test['embeddings'][embedding_pos]
        del test, embedding_pos

        #creating a new pickle file
        with open('features.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        del embeddings

        df1 = df1.drop(["embedding_pos"])
        df1.rename('new_emb_pos','embedding_pos')


        return df1


def clean_node_edge_list(df1,file2:str,file3:str):
        
        #opening grantreduced and edgelist
        df1 
        df2 = vx.open(file2)

        # Remove isolates
        df1.rename('patnum','sender')
        df1.rename('embedding_pos','sender_pos')
        df1.rename('pub_year','event_year')
        df1.rename('pub_date_days','event_day')
        df_join = df2.join(df1,on='sender',how='inner',allow_duplication=True)

        print('ok1')
        # Removes anything before the oldest year in df1
        df1.rename('sender','receiver')
        df1.rename('sender_pos','receiver_pos')
        df1.rename('event_year','rec_pub_year')
        df1.rename('event_day','rec_pub_day')
        df_join = df_join.join(df1,on='receiver',how='inner',allow_duplication=True)

        del df1,df2

        print('ok2')
        df_join = remove_duplicates(df_join,['sender_pos','receiver_pos'])
        df_join=df_join[df_join['event_day']>=df_join['rec_pub_day']]
        df_join.extract()
        print('ok3')

        # emb_to_keep is the ordered and unique patent position extracted from df2 after cleaning
        
        # list1 = df_join["sender_pos"].values.tolist()
        # list2 = df_join["receiver_pos"].values.tolist()

        list3 = df_join["sender_pos"].values.tolist() + df_join["receiver_pos"].values.tolist()
        emb_to_keep = np.unique(np.array(list3))
        print(emb_to_keep)
        del list3

        pkl_file = open(file3,'rb')
        test=np.array(pickle.load(pkl_file))
        pkl_file.close()
        print(test)

        # here I remove the embeddings that are no longer needed (because the patent was removed in df2)
        embeddings=test[emb_to_keep]
        del test

        with open('features2.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
        print(len(embeddings))
        del embeddings


        print('ok4')

        d=dict(zip(emb_to_keep, np.array(range(len(emb_to_keep)))))


        # df_join['sender_pos']=df_join['sender_pos'].map(d)
        # df_join['receiver_pos']=df_join['receiver_pos'].map(d)
        print('ok5')

        return df_join, d

def update_emb_positions(df_join,d):
        df_join['sender_pos']=df_join['sender_pos'].map(d)
        df_join['receiver_pos']=df_join['receiver_pos'].map(d)

        return df_join
        
def update_node_list (df_join):
        df_sender=df_join[['sender','sender_pos','event_year','event_day']]
        df_receiver=df_join[['receiver','receiver_pos','rec_pub_year','rec_pub_day']]
        

        del df_join

        df_sender=vaex_drop_duplicates(df_sender)
        rename_dict = {'sender': 'Pat_num', 'sender_pos': 'Position'}
        for old_name, new_name in rename_dict.items():
              df_sender.rename(old_name, new_name)  

        df_receiver=vaex_drop_duplicates(df_receiver)
        rename_dict = {'receiver': 'Pat_num', 'receiver_pos': 'Position', 'rec_pub_year':'event_year', 'rec_pub_day':'event_day'}
        
        for old_name, new_name in rename_dict.items():
              df_receiver.rename(old_name, new_name) 
                                         
        df = df_sender.concat(df_receiver)

        del df_sender,df_receiver

        df=vaex_drop_duplicates(df)
        print(len(df))
        return df
        
def in_degree_in_nodelst(df_join,df_node,In_degree:str):
        df3=df_join['receiver_pos'].value_counts().to_frame()
        df3=df3.reset_index(drop=False)
        df3.columns=['Position',In_degree]
        test = vx.from_pandas(df3)
        df_node2 = df_node.join(test,on='Position',how='left',allow_duplication=True)
        df_node2[In_degree]=df_node2[In_degree].fillna(0)

        return df_node2
      

def vaex_drop_duplicates(df, columns=None):
    """Return a :class:`DataFrame` object with no duplicates in the given columns.
    .. warning:: The resulting dataframe will be in memory, use with caution.
    :param columns: Column or list of column to remove duplicates by, default to all columns.
    :return: :class:`DataFrame` object with duplicates filtered away.
    """
    if columns is None:
        columns = df.get_column_names()
    if type(columns) is str:
        columns = [columns]
    return df.groupby(columns, agg={'__hidden_count': vx.agg.count()}).drop('__hidden_count').extract()