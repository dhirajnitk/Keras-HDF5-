# Keras-HDF5 Read/Write With h5py-cache 

### write_generator: Writing resized image features to HDF5 File
``` 
    import h5py_cache as h5c
    import h5py as h5
    datagen = ImageDataGenerator()
    #Build a fake model
    model = Sequential()
    generator = datagen.flow_from_directory(
        full_data_dir,
        target_size=(img_height,img_width),
        )
    shape = (generator.samples, img_width,img_height,3 )
    chunk_shape=(1, 100,100,3)
    total_mem_usage,dividing_factor = min(np.prod(shape)*4,1024**2*15000),4
    f1_all = h5c.File(all_features_hdf5, 'w',chunk_cache_mem_size=total_mem_usage//dividing_factor)
    f1_label = h5.File(all_labels_hdf5, 'w')
    d1_all = f1_all.create_dataset('data', shape ,dtype='float32',chunks=chunk_shape,compression="lzf")
    d1_label = f1_label.create_dataset('data', (generator.samples,len(generator.table_pd.columns)) ,dtype='float32')
    # Writes Raw features if ImageDataGenerator is blank
    model.write_generator(generator, 
                          steps = int(ceil(generator.samples/ batch_size)),
                          max_queue_size=10,
                          workers=4,
                          d_set= d1_all,
                          label_set =d1_label)
    f1_all.close()
    f1_label.close()
    
 ```
 ###  write_predict_generator :  Writing bottlneck features to HDF5 File
 
 ```
    
    #With a genuine model, you can write the bottleneck features to files after flow_from_directory
    # h5py_file and h5py_label  are corresponding file names for train and label .h5 files. 
    model.write_predict_generator(generator, 
                                  steps=generator.samples//batch_size,
                                  max_queue_size=10,
                                  h5py_file = all_features_hdf5,
                                  h5py_label =all_labels_hdf5)
```
### flow_hdf5 :  Read from HDF5 File using a sequence Iterator(HDF5MatrixCacheIterator).
 
```
  # Apply_gen_transform is required if  Features are to be transformed and standardized. Default is False
    datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                 validation_split=0.2,
                                 apply_gen_transform= True)
    f1_trainvalidation = h5c.File(all_features_hdf5, 'r',chunk_cache_mem_size=total_mem_usage//dividing_factor)
    f1_label = h5.File(all_labels_hdf5, 'r')
    # Shuffle false ensures  training or validation sequence wont be shuffled. They will be in order and faster to fetch
    # fit_generaator shuffle flag only shuffles the batches. Default for Shuffle is False
    train_generator = datagen.flow_hdf5( f1_trainvalidation['data'],
                                         f1_label['data'],
                                         subset = 'training',
                                         shuffle=False)
    validation_generator = datagen.flow_hdf5(f1_trainvalidation['data'],
                                            f1_label['data'],
                                            subset = 'validation',
                                            shuffle=False)
       
