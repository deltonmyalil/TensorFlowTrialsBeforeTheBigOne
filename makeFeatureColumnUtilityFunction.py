## Use this function to make a feature column for a desne NN Class/regr
## Assuming to havev imported tensorflow as tf already in the 
## Calling script
## made by Delton


## allColumns is a list of all columns in the dataset in string form
## numericCOlumns is a list of numeric columns in the dataset, in order
## categoricalColumns is a list of all cat Columns in the dataset in order
## categoricalColumnsDimension is the list of integers of the number of categories in each cat column present in the categoricalColumns list

## The columns have to filled in the order it originally is in the dataframe

## Use assert statement in the calling script/notebook cell to prevent tf errors when model training
def getFeatureColumns(allColumns, numericColumns, categoricalColumns, 
                      categoricalColumnsDimension):
    # Make an empty list of feature columns to be filled iteratively
    feat_cols = []
    for column in allColumns:
        if column in numericColumns:
            feat_cols.append(tf.feature_column.numeric_column(column))
        elif column in categoricalColumns:
            dim = categoricalColumnsDimension[categoricalColumns.index(column)]
            tempCatCol = tf.feature_column.categorical_column_with_hash_bucket(column,
                                                                        hash_bucket_size=dim)
            # Do this only when using DNNClassr or DNNRegr
            embeddedCatCol = tf.feature_column.embedding_column(tempCatCol, 
                                                               dimension=dim)
            feat_cols.append(embeddedCatCol)
    return feat_cols
