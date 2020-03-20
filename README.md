# Deep features for image retrieval assignament (Semana 6)
Jupyter notebook corresponding to the assignament of the Unit: "Neural networks: Learning very non-linear features" in the Coursera "Machine Learning Foundations: A Case Study Approach" course.


```python
import turicreate
image_data = turicreate.SFrame('image_train_data')
image_test = turicreate.SFrame('image_test_data')
```

## Task 1: Compute summary statistics of the data
Compute the sketch summary of the "label" column and interpret the results


```python
sketch = turicreate.Sketch(image_data['label'])
```


```python
sketch
```




    
    +------------------+-------+----------+
    |       item       | value | is exact |
    +------------------+-------+----------+
    |      Length      |  2005 |   Yes    |
    | # Missing Values |   0   |   Yes    |
    | # unique values  |   4   |    No    |
    +------------------+-------+----------+
    
    Most frequent items:
    +------------+-------+
    |   value    | count |
    +------------+-------+
    |    cat     |  509  |
    |    dog     |  509  |
    | automobile |  509  |
    |    bird    |  478  |
    +------------+-------+




## Task 2: Create category-specific image retrieval models


```python
dogs = image_data.filter_by('dog','label')
```


```python
turicreate.Sketch(dogs['label'])
```




    
    +------------------+-------+----------+
    |       item       | value | is exact |
    +------------------+-------+----------+
    |      Length      |  509  |   Yes    |
    | # Missing Values |   0   |   Yes    |
    | # unique values  |   1   |    No    |
    +------------------+-------+----------+
    
    Most frequent items:
    +-------+-------+
    | value | count |
    +-------+-------+
    |  dog  |  509  |
    +-------+-------+





```python
cats = image_data.filter_by('cat','label')
automobiles = image_data.filter_by('automobile','label')
birds = image_data.filter_by('bird','label')
```


```python
dog_model = turicreate.nearest_neighbors.create(dogs,
                                               features = ['deep_features'],
                                               label = 'id')
cat_model = turicreate.nearest_neighbors.create(cats,
                                               features = ['deep_features'],
                                               label = 'id')
automobile_model = turicreate.nearest_neighbors.create(automobiles,
                                               features = ['deep_features'],
                                               label = 'id')
bird_model = turicreate.nearest_neighbors.create(birds,
                                               features = ['deep_features'],
                                               label = 'id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



```python
cat = image_test[0:1]
cat['image'].explore()
```


<html lang="en">                                                     <head>                                                               <style>                                                              .sframe {                                                            font-size: 12px;                                                   font-family: HelveticaNeue;                                        border: 1px solid silver;                                        }                                                                  .sframe thead th {                                                   background: #F7F7F7;                                               font-family: HelveticaNeue-Medium;                                 font-size: 14px;                                                   line-height: 16.8px;                                               padding-top: 16px;                                                 padding-bottom: 16px;                                              padding-left: 10px;                                                padding-right: 38px;                                               border-top: 1px solid #E9E9E9;                                     border-bottom: 1px solid #E9E9E9;                                  white-space: nowrap;                                               overflow: hidden;                                                  text-overflow:ellipsis;                                            text-align:center;                                                 font-weight:normal;                                              }                                                                  .sframe tbody th {                                                   background: #FFFFFF;                                               text-align:left;                                                   font-weight:normal;                                                border-right: 1px solid #E9E9E9;                                 }                                                                  .sframe td {                                                         background: #FFFFFF;                                               padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr {                                                         padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr:hover {                                                   background: silver;                                              },                                                               </style>                                                         </head>                                                            <body>                                                               <h1>  </h1>                                             <table border="1" class="dataframe sframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SArray</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td><img src="data:image/png;base64,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"/></td>
    </tr>
  </tbody>
</table>                          </body>                                                          </html>



```python
cat_model.query(cat)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 12.984ms     |</pre>



<pre>| Done         |         | 100         | 134.529ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">16289</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.62371920804245</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45646</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.00687992842462</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32139</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.52008134363789</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">25713</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.754850252057054</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">331</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.87312281675268</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
cats[cats['id'] == 16289]['image'].explore()
```


<html lang="en">                                                     <head>                                                               <style>                                                              .sframe {                                                            font-size: 12px;                                                   font-family: HelveticaNeue;                                        border: 1px solid silver;                                        }                                                                  .sframe thead th {                                                   background: #F7F7F7;                                               font-family: HelveticaNeue-Medium;                                 font-size: 14px;                                                   line-height: 16.8px;                                               padding-top: 16px;                                                 padding-bottom: 16px;                                              padding-left: 10px;                                                padding-right: 38px;                                               border-top: 1px solid #E9E9E9;                                     border-bottom: 1px solid #E9E9E9;                                  white-space: nowrap;                                               overflow: hidden;                                                  text-overflow:ellipsis;                                            text-align:center;                                                 font-weight:normal;                                              }                                                                  .sframe tbody th {                                                   background: #FFFFFF;                                               text-align:left;                                                   font-weight:normal;                                                border-right: 1px solid #E9E9E9;                                 }                                                                  .sframe td {                                                         background: #FFFFFF;                                               padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr {                                                         padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr:hover {                                                   background: silver;                                              },                                                               </style>                                                         </head>                                                            <body>                                                               <h1>  </h1>                                             <table border="1" class="dataframe sframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SArray</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td><img src="data:image/png;base64,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"/></td>
    </tr>
  </tbody>
</table>                          </body>                                                          </html>



```python
dog_model.query(cat)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.196464    | 16.784ms     |</pre>



<pre>| Done         |         | 100         | 199.468ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">16976</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.464262878423774</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13387</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.56668321685285</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35867</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.60472670789396</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">44603</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.70655851529755</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6094</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.511325490739715</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
dogs[dogs['id'] == 16976]['image'].explore()
```


<html lang="en">                                                     <head>                                                               <style>                                                              .sframe {                                                            font-size: 12px;                                                   font-family: HelveticaNeue;                                        border: 1px solid silver;                                        }                                                                  .sframe thead th {                                                   background: #F7F7F7;                                               font-family: HelveticaNeue-Medium;                                 font-size: 14px;                                                   line-height: 16.8px;                                               padding-top: 16px;                                                 padding-bottom: 16px;                                              padding-left: 10px;                                                padding-right: 38px;                                               border-top: 1px solid #E9E9E9;                                     border-bottom: 1px solid #E9E9E9;                                  white-space: nowrap;                                               overflow: hidden;                                                  text-overflow:ellipsis;                                            text-align:center;                                                 font-weight:normal;                                              }                                                                  .sframe tbody th {                                                   background: #FFFFFF;                                               text-align:left;                                                   font-weight:normal;                                                border-right: 1px solid #E9E9E9;                                 }                                                                  .sframe td {                                                         background: #FFFFFF;                                               padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr {                                                         padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr:hover {                                                   background: silver;                                              },                                                               </style>                                                         </head>                                                            <body>                                                               <h1>  </h1>                                             <table border="1" class="dataframe sframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SArray</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td><img src="data:image/png;base64,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"/></td>
    </tr>
  </tbody>
</table>                          </body>                                                          </html>


## Task 3: Try a simple example of nearest-neighbors classification


```python
image_test_cat = image_test.filter_by('cat','label')
image_test_dog = image_test.filter_by('dog','label')
```


```python
test_dog_model = turicreate.nearest_neighbors.create(image_test_dog,
                                              features = ['deep_features'],
                                               label = 'id')
test_cat_model = turicreate.nearest_neighbors.create(image_test_cat,
                                               features = ['deep_features'],
                                               label = 'id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



```python
first_image = image_test[0:1]
```


```python
test_cat_model.query(first_image)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.1         | 23.749ms     |</pre>



<pre>| Done         |         | 100         | 220.773ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">586</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">31.738566258970284</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7980</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.31915884686147</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2269</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.82505477406374</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4674</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.187010167130694</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
test_dog_model.query(first_image)
```


<pre>Starting pairwise querying.</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 0            | 1       | 0.1         | 20.62ms      |</pre>



<pre>| Done         |         | 100         | 236.736ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>





<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6122</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.19828434885113</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9657</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.39238619955533</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4551</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.58533760444701</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">640</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.005841510711605</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6924</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.23568768295601</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
    </tr>
</table>
[5 rows x 4 columns]<br/>
</div>




```python
first_image['image'].explore()
```


<html lang="en">                                                     <head>                                                               <style>                                                              .sframe {                                                            font-size: 12px;                                                   font-family: HelveticaNeue;                                        border: 1px solid silver;                                        }                                                                  .sframe thead th {                                                   background: #F7F7F7;                                               font-family: HelveticaNeue-Medium;                                 font-size: 14px;                                                   line-height: 16.8px;                                               padding-top: 16px;                                                 padding-bottom: 16px;                                              padding-left: 10px;                                                padding-right: 38px;                                               border-top: 1px solid #E9E9E9;                                     border-bottom: 1px solid #E9E9E9;                                  white-space: nowrap;                                               overflow: hidden;                                                  text-overflow:ellipsis;                                            text-align:center;                                                 font-weight:normal;                                              }                                                                  .sframe tbody th {                                                   background: #FFFFFF;                                               text-align:left;                                                   font-weight:normal;                                                border-right: 1px solid #E9E9E9;                                 }                                                                  .sframe td {                                                         background: #FFFFFF;                                               padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr {                                                         padding-left: 10px;                                                padding-right: 38px;                                               padding-top: 14px;                                                 padding-bottom: 14px;                                              border-bottom: 1px solid #E9E9E9;                                  max-height: 0px;                                                   transition: max-height 5s ease-out;                                vertical-align: middle;                                            font-family: HelveticaNeue;                                        font-size: 12px;                                                   line-height: 16.8px;                                               background: #FFFFFF;                                             }                                                                  .sframe tr:hover {                                                   background: silver;                                              },                                                               </style>                                                         </head>                                                            <body>                                                               <h1>  </h1>                                             <table border="1" class="dataframe sframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SArray</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td><img src="data:image/png;base64,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"/></td>
    </tr>
  </tbody>
</table>                          </body>                                                          </html>



```python

```

## Task 4: Compute nearest neighbors accuracy


```python
image_test_bird = image_test.filter_by('bird','label')
image_test_automobile = image_test.filter_by('automobile','label')
```


```python
test_bird_model = turicreate.nearest_neighbors.create(image_test_bird,
                                              features = ['deep_features'],
                                               label = 'id')
test_automobile_model = turicreate.nearest_neighbors.create(image_test_automobile,
                                               features = ['deep_features'],
                                               label = 'id')
```


<pre>Starting brute force nearest neighbors model training.</pre>



<pre>Starting brute force nearest neighbors model training.</pre>



```python
dog_cat_neighbors = cat_model.query(image_test_dog,k=1)
dog_bird_neighbors = bird_model.query(image_test_dog,k=1)
dog_automobile_neighbors = automobile_model.query(image_test_dog,k=1)
dog_dog_neighbors = dog_model.query(image_test_dog,k=1)
```


<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 4348</pre>



<pre>number of reference data blocks: 2</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 254000  | 49.9018     | 625.604ms    |</pre>



<pre>| Done         | 509000  | 100         | 663.887ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 4348</pre>



<pre>number of reference data blocks: 2</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 239000  | 50          | 601.052ms    |</pre>



<pre>| Done         | 478000  | 100         | 622.992ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 4348</pre>



<pre>number of reference data blocks: 2</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 254000  | 49.9018     | 642.406ms    |</pre>



<pre>| Done         | 509000  | 100         | 659.626ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>Starting blockwise querying.</pre>



<pre>max rows per data block: 4348</pre>



<pre>number of reference data blocks: 2</pre>



<pre>number of query data blocks: 1</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| Query points | # Pairs | % Complete. | Elapsed Time |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



<pre>| 1000         | 254000  | 49.9018     | 618.119ms    |</pre>



<pre>| Done         | 509000  | 100         | 640.432ms    |</pre>



<pre>+--------------+---------+-------------+--------------+</pre>



```python
dog_cat_neighbors
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.419607706754384</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">30606</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.83532688735542</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5545</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.97634108541546</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">19631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.575007291446106</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7493</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.77882479101661</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47044</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.11715782924591</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13918</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.60958309132649</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10981</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.90368673062214</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45456</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.067470016821176</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">44673</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.72587329506032</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[1000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
dog_dog_neighbors
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">query_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">reference_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">distance</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rank</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">49803</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.47735903726335</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5755</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32.84584956840554</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20715</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.03970731890584</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13387</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.90103276968193</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12089</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.484925090925636</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6094</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.94516534398124</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3431</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.095727834463545</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6184</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.76961310322034</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2167</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.10891446032838</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7776</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.242283258453455</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[1000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
dog_distances = turicreate.SFrame({'dog-dog':dog_dog_neighbors['distance'],
                                   'dog-cat':dog_cat_neighbors['distance'],
                                  'dog-bird':dog_bird_neighbors['distance'],
                                  'dog-automobile':dog_automobile_neighbors['distance']})
```


```python
dog_distances.show()
```


<pre>Materializing SFrame</pre>



<html>                 <body>                     <iframe style="border:0;margin:0" width="1000" height="1500" srcdoc='<html lang="en">                         <head>                             <script src="https://cdnjs.cloudflare.com/ajax/libs/vega/5.4.0/vega.js"></script>                             <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-embed/4.0.0/vega-embed.js"></script>                             <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-tooltip/0.5.1/vega-tooltip.min.js"></script>                             <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/vega-tooltip/0.5.1/vega-tooltip.min.css">                             <style>                             .vega-actions > a{                                 color:white;                                 text-decoration: none;                                 font-family: "Arial";                                 cursor:pointer;                                 padding:5px;                                 background:#AAAAAA;                                 border-radius:4px;                                 padding-left:10px;                                 padding-right:10px;                                 margin-right:5px;                             }                             .vega-actions{                                 margin-top:20px;                                 text-align:center                             }                            .vega-actions > a{                                 background:#999999;                            }                             </style>                         </head>                         <body>                             <div id="vis">                             </div>                             <script>                                 var vega_json = "{\"$schema\": \"https://vega.github.io/schema/vega/v4.json\", \"metadata\": {\"bubbleOpts\": {\"showAllFields\": false, \"fields\": [{\"field\": \"left\"}, {\"field\": \"right\"}, {\"field\": \"count\"}, {\"field\": \"label\"}]}}, \"width\": 800, \"height\": 1280, \"padding\": 8, \"data\": [{\"name\": \"pts_store\"}, {\"name\": \"source_2\", \"values\": [{\"a\": 0, \"title\": \"dog-automobile\", \"num_row\": 1000, \"type\": \"float\", \"num_unique\": 1000, \"num_missing\": 0, \"mean\": 43.020187, \"min\": 33.514852, \"max\": 56.371804, \"median\": 42.874644, \"stdev\": 3.291573, \"numeric\": [{\"left\": 33.3035, \"right\": 34.4682, \"count\": 1}, {\"left\": 34.4682, \"right\": 35.6329, \"count\": 4}, {\"left\": 35.6329, \"right\": 36.7976, \"count\": 12}, {\"left\": 36.7976, \"right\": 37.9623, \"count\": 26}, {\"left\": 37.9623, \"right\": 39.1271, \"count\": 71}, {\"left\": 39.1271, \"right\": 40.2918, \"count\": 100}, {\"left\": 40.2918, \"right\": 41.4565, \"count\": 122}, {\"left\": 41.4565, \"right\": 42.6212, \"count\": 139}, {\"left\": 42.6212, \"right\": 43.7859, \"count\": 134}, {\"left\": 43.7859, \"right\": 44.9507, \"count\": 126}, {\"left\": 44.9507, \"right\": 46.1154, \"count\": 102}, {\"left\": 46.1154, \"right\": 47.2801, \"count\": 62}, {\"left\": 47.2801, \"right\": 48.4448, \"count\": 43}, {\"left\": 48.4448, \"right\": 49.6095, \"count\": 30}, {\"left\": 49.6095, \"right\": 50.7742, \"count\": 13}, {\"left\": 50.7742, \"right\": 51.939, \"count\": 4}, {\"left\": 51.939, \"right\": 53.1037, \"count\": 6}, {\"left\": 53.1037, \"right\": 54.2684, \"count\": 2}, {\"left\": 54.2684, \"right\": 55.4331, \"count\": 2}, {\"left\": 55.4331, \"right\": 56.5978, \"count\": 1}, {\"start\": 33.3035, \"stop\": 56.5978, \"step\": 1.16472}], \"categorical\": []}, {\"a\": 1, \"title\": \"dog-bird\", \"num_row\": 1000, \"type\": \"float\", \"num_unique\": 1000, \"num_missing\": 0, \"mean\": 39.197041, \"min\": 28.316228, \"max\": 51.760868, \"median\": 39.092149, \"stdev\": 3.489211, \"numeric\": [{\"left\": 28.2479, \"right\": 29.4367, \"count\": 2}, {\"left\": 29.4367, \"right\": 30.6256, \"count\": 4}, {\"left\": 30.6256, \"right\": 31.8145, \"count\": 4}, {\"left\": 31.8145, \"right\": 33.0033, \"count\": 19}, {\"left\": 33.0033, \"right\": 34.1922, \"count\": 35}, {\"left\": 34.1922, \"right\": 35.381, \"count\": 63}, {\"left\": 35.381, \"right\": 36.5699, \"count\": 88}, {\"left\": 36.5699, \"right\": 37.7588, \"count\": 120}, {\"left\": 37.7588, \"right\": 38.9476, \"count\": 140}, {\"left\": 38.9476, \"right\": 40.1365, \"count\": 132}, {\"left\": 40.1365, \"right\": 41.3253, \"count\": 142}, {\"left\": 41.3253, \"right\": 42.5142, \"count\": 92}, {\"left\": 42.5142, \"right\": 43.703, \"count\": 45}, {\"left\": 43.703, \"right\": 44.8919, \"count\": 46}, {\"left\": 44.8919, \"right\": 46.0808, \"count\": 42}, {\"left\": 46.0808, \"right\": 47.2696, \"count\": 10}, {\"left\": 47.2696, \"right\": 48.4585, \"count\": 7}, {\"left\": 48.4585, \"right\": 49.6473, \"count\": 6}, {\"left\": 49.6473, \"right\": 50.8362, \"count\": 2}, {\"left\": 50.8362, \"right\": 52.0251, \"count\": 1}, {\"start\": 28.2479, \"stop\": 52.0251, \"step\": 1.18886}], \"categorical\": []}, {\"a\": 2, \"title\": \"dog-cat\", \"num_row\": 1000, \"type\": \"float\", \"num_unique\": 1000, \"num_missing\": 0, \"mean\": 37.003045, \"min\": 28.177487, \"max\": 52.728657, \"median\": 36.671667, \"stdev\": 3.72093, \"numeric\": [{\"left\": 28.0805, \"right\": 29.3174, \"count\": 7}, {\"left\": 29.3174, \"right\": 30.5542, \"count\": 11}, {\"left\": 30.5542, \"right\": 31.7911, \"count\": 49}, {\"left\": 31.7911, \"right\": 33.0279, \"count\": 75}, {\"left\": 33.0279, \"right\": 34.2648, \"count\": 102}, {\"left\": 34.2648, \"right\": 35.5016, \"count\": 122}, {\"left\": 35.5016, \"right\": 36.7385, \"count\": 141}, {\"left\": 36.7385, \"right\": 37.9753, \"count\": 126}, {\"left\": 37.9753, \"right\": 39.2122, \"count\": 102}, {\"left\": 39.2122, \"right\": 40.449, \"count\": 86}, {\"left\": 40.449, \"right\": 41.6859, \"count\": 64}, {\"left\": 41.6859, \"right\": 42.9227, \"count\": 52}, {\"left\": 42.9227, \"right\": 44.1596, \"count\": 29}, {\"left\": 44.1596, \"right\": 45.3964, \"count\": 14}, {\"left\": 45.3964, \"right\": 46.6333, \"count\": 7}, {\"left\": 46.6333, \"right\": 47.8701, \"count\": 7}, {\"left\": 47.8701, \"right\": 49.107, \"count\": 3}, {\"left\": 49.107, \"right\": 50.3438, \"count\": 2}, {\"left\": 50.3438, \"right\": 51.5807, \"count\": 0}, {\"left\": 51.5807, \"right\": 52.8175, \"count\": 1}, {\"start\": 28.0805, \"stop\": 52.8175, \"step\": 1.23685}], \"categorical\": []}, {\"a\": 3, \"title\": \"dog-dog\", \"num_row\": 1000, \"type\": \"float\", \"num_unique\": 1000, \"num_missing\": 0, \"mean\": 35.546373, \"min\": 19.671404, \"max\": 51.18219, \"median\": 35.230753, \"stdev\": 3.640294, \"numeric\": [{\"left\": 19.6625, \"right\": 21.2387, \"count\": 1}, {\"left\": 21.2387, \"right\": 22.815, \"count\": 1}, {\"left\": 22.815, \"right\": 24.3912, \"count\": 0}, {\"left\": 24.3912, \"right\": 25.9674, \"count\": 3}, {\"left\": 25.9674, \"right\": 27.5437, \"count\": 6}, {\"left\": 27.5437, \"right\": 29.1199, \"count\": 13}, {\"left\": 29.1199, \"right\": 30.6962, \"count\": 61}, {\"left\": 30.6962, \"right\": 32.2724, \"count\": 77}, {\"left\": 32.2724, \"right\": 33.8487, \"count\": 163}, {\"left\": 33.8487, \"right\": 35.4249, \"count\": 199}, {\"left\": 35.4249, \"right\": 37.0012, \"count\": 158}, {\"left\": 37.0012, \"right\": 38.5774, \"count\": 128}, {\"left\": 38.5774, \"right\": 40.1537, \"count\": 87}, {\"left\": 40.1537, \"right\": 41.7299, \"count\": 49}, {\"left\": 41.7299, \"right\": 43.3062, \"count\": 30}, {\"left\": 43.3062, \"right\": 44.8824, \"count\": 12}, {\"left\": 44.8824, \"right\": 46.4587, \"count\": 6}, {\"left\": 46.4587, \"right\": 48.0349, \"count\": 4}, {\"left\": 48.0349, \"right\": 49.6112, \"count\": 1}, {\"left\": 49.6112, \"right\": 51.1874, \"count\": 1}, {\"start\": 19.6625, \"stop\": 51.1874, \"step\": 1.57625}], \"categorical\": []}]}, {\"name\": \"data_2\", \"source\": \"source_2\", \"transform\": [{\"type\": \"formula\", \"expr\": \"20\", \"as\": \"c_x_axis_back\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+66\", \"as\": \"c_main_background\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+43\", \"as\": \"c_top_bar\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+59\", \"as\": \"c_top_title\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+58\", \"as\": \"c_top_type\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+178\", \"as\": \"c_rule\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+106\", \"as\": \"c_num_rows\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+130\", \"as\": \"c_num_unique\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+154\", \"as\": \"c_missing\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+105\", \"as\": \"c_num_rows_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+130\", \"as\": \"c_num_unique_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+154\", \"as\": \"c_missing_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+195\", \"as\": \"c_frequent_items\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+218\", \"as\": \"c_first_item\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+235\", \"as\": \"c_second_item\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+252\", \"as\": \"c_third_item\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+269\", \"as\": \"c_fourth_item\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+286\", \"as\": \"c_fifth_item\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+200\", \"as\": \"c_mean\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+220\", \"as\": \"c_min\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+240\", \"as\": \"c_max\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+260\", \"as\": \"c_median\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+280\", \"as\": \"c_stdev\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+198\", \"as\": \"c_mean_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+218\", \"as\": \"c_min_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+238\", \"as\": \"c_max_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+258\", \"as\": \"c_median_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+278\", \"as\": \"c_stdev_val\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+106\", \"as\": \"graph_offset\"}, {\"type\": \"formula\", \"expr\": \"toNumber(datum[\\\"a\\\"])*300+132\", \"as\": \"graph_offset_categorical\"}, {\"type\": \"formula\", \"expr\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?false:true\", \"as\": \"c_clip_val\"}, {\"type\": \"formula\", \"expr\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?250:0\", \"as\": \"c_width_numeric_val\"}, {\"type\": \"formula\", \"expr\": \"(toString(datum[\\\"type\\\"]) == \\\"str\\\")?false:true\", \"as\": \"c_clip_val_cat\"}, {\"type\": \"formula\", \"expr\": \"(toString(datum[\\\"type\\\"]) == \\\"str\\\")?250:0\", \"as\": \"c_width_numeric_val_cat\"}]}], \"marks\": [{\"encode\": {\"enter\": {\"x\": {\"value\": 0}, \"width\": {\"value\": 734}, \"y\": {\"value\": 0}, \"height\": {\"value\": 366}, \"clip\": {\"value\": 0}, \"fill\": {\"value\": \"#ffffff\"}, \"fillOpacity\": {\"value\": 0}, \"stroke\": {\"value\": \"#000000\"}, \"strokeWidth\": {\"value\": 0}}}, \"marks\": [{\"encode\": {\"enter\": {\"x\": {\"value\": 0}, \"width\": {\"value\": 734}, \"y\": {\"value\": 0}, \"height\": {\"value\": 366}, \"clip\": {\"value\": 0}, \"fill\": {\"value\": \"#ffffff\"}, \"fillOpacity\": {\"value\": 0}, \"stroke\": {\"value\": \"#000000\"}, \"strokeWidth\": {\"value\": 0}}}, \"scales\": [], \"axes\": [], \"marks\": [{\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 33}, \"width\": {\"value\": 700}, \"y\": {\"value\": 66}, \"height\": {\"value\": 250}, \"fill\": {\"value\": \"#FEFEFE\"}, \"fillOpacity\": {\"value\": 1}, \"stroke\": {\"value\": \"#DEDEDE\"}, \"strokeWidth\": {\"value\": 0.5}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]\"}, \"y\": {\"field\": \"c_main_background\"}}}, \"type\": \"rect\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 33}, \"width\": {\"value\": 700}, \"y\": {\"value\": 43}, \"height\": {\"value\": 30}, \"fill\": {\"value\": \"#F5F5F5\"}, \"fillOpacity\": {\"value\": 1}, \"stroke\": {\"value\": \"#DEDEDE\"}, \"strokeWidth\": {\"value\": 0.5}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]\"}, \"y\": {\"field\": \"c_top_bar\"}}}, \"type\": \"rect\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 720}, \"y\": {\"value\": 58}, \"text\": {\"signal\": \"&apos;&apos;+datum[\\\"type\\\"]\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#595859\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+687\"}, \"y\": {\"field\": \"c_top_type\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 44}, \"y\": {\"value\": 59}, \"text\": {\"signal\": \"&apos;&apos;+datum[\\\"title\\\"]\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 15}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#9B9B9B\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+11\"}, \"y\": {\"field\": \"c_top_title\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 178}, \"stroke\": {\"value\": \"#EDEDEB\"}, \"strokeWidth\": {\"value\": 1}, \"strokeCap\": {\"value\": \"butt\"}, \"x2\": {\"value\": 720}, \"y2\": {\"value\": 178}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"x2\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+687\"}, \"y\": {\"field\": \"c_rule\"}, \"y2\": {\"field\": \"c_rule\"}}}, \"type\": \"rule\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 106}, \"text\": {\"value\": \"Num. Rows:\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_num_rows\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 130}, \"text\": {\"value\": \"Num. Unique:\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_num_unique\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 154}, \"text\": {\"value\": \"Missing:\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_missing\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 105}, \"text\": {\"signal\": \"toString(format(datum[\\\"num_row\\\"], \\\",\\\"))\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#5A5A5A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_num_rows_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 130}, \"text\": {\"signal\": \"toString(format(datum[\\\"num_unique\\\"], \\\",\\\"))\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#5A5A5A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_num_unique_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 154}, \"text\": {\"signal\": \"toString(format(datum[\\\"num_missing\\\"], \\\",\\\"))\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 12}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#5A5A5A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_missing_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"str\\\")? \\\"Frequent Items\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_frequent_items\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 520}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 1) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][0][\\\"label\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+487\"}, \"y\": {\"field\": \"c_first_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 520}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 2) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][1][\\\"label\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+487\"}, \"y\": {\"field\": \"c_second_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 520}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 3) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][2][\\\"label\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+487\"}, \"y\": {\"field\": \"c_third_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 520}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 4) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][3][\\\"label\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+487\"}, \"y\": {\"field\": \"c_fourth_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 520}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 5) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][4][\\\"label\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+487\"}, \"y\": {\"field\": \"c_fifth_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 1) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][0][\\\"count\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#7A7A7A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_first_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 2) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][1][\\\"count\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#7A7A7A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_second_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 3) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][2][\\\"count\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#7A7A7A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_third_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 4) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][3][\\\"count\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#7A7A7A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_fourth_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"((datum[\\\"categorical\\\"].length >= 5) &amp;&amp; (toString(datum[\\\"type\\\"]) == \\\"str\\\"))? toString(datum[\\\"categorical\\\"][4][\\\"count\\\"]):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#7A7A7A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_fifth_item\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 200}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")? \\\"Mean:\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"clip\": {\"value\": true}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_mean\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 220}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")? \\\"Min:\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_min\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 240}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")? \\\"Max:\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_max\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 260}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")? \\\"Median:\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_median\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 500}, \"y\": {\"value\": 280}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")? \\\"St. Dev:\\\":\\\"\\\"\"}, \"align\": {\"value\": \"left\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 11}, \"fontWeight\": {\"value\": \"bold\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#4A4A4A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+467\"}, \"y\": {\"field\": \"c_stdev\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 198}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?toString(format(datum[\\\"mean\\\"], \\\",\\\")):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#6A6A6A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_mean_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 218}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?toString(format(datum[\\\"min\\\"], \\\",\\\")):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#6A6A6A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_min_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 238}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?toString(format(datum[\\\"max\\\"], \\\",\\\")):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#6A6A6A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_max_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 258}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?toString(format(datum[\\\"median\\\"], \\\",\\\")):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#6A6A6A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_median_val\"}}}, \"type\": \"text\"}, {\"from\": {\"data\": \"data_2\"}, \"encode\": {\"enter\": {\"x\": {\"value\": 700}, \"y\": {\"value\": 278}, \"text\": {\"signal\": \"(toString(datum[\\\"type\\\"]) == \\\"integer\\\" || toString(datum[\\\"type\\\"]) == \\\"float\\\")?toString(format(datum[\\\"stdev\\\"], \\\",\\\")):\\\"\\\"\"}, \"align\": {\"value\": \"right\"}, \"baseline\": {\"value\": \"middle\"}, \"dx\": {\"value\": 0, \"offset\": 0}, \"dy\": {\"value\": 0, \"offset\": 0}, \"angle\": {\"value\": 0}, \"font\": {\"value\": \"AvenirNext-Medium\"}, \"fontSize\": {\"value\": 10}, \"fontWeight\": {\"value\": \"normal\"}, \"fontStyle\": {\"value\": \"normal\"}, \"fill\": {\"value\": \"#6A6A6A\"}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+667\"}, \"y\": {\"field\": \"c_stdev_val\"}}}, \"type\": \"text\"}, {\"from\": {\"facet\": {\"name\": \"new_data\", \"data\": \"data_2\", \"field\": \"numeric\"}}, \"encode\": {\"enter\": {\"x\": {\"value\": 120}, \"width\": {\"value\": 250}, \"y\": {\"field\": \"graph_offset\"}, \"height\": {\"value\": 150}, \"fill\": {\"value\": \"#ffffff\"}, \"fillOpacity\": {\"value\": 0}, \"stroke\": {\"value\": \"#000000\"}, \"strokeWidth\": {\"value\": 0}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+87\"}, \"clip\": {\"field\": \"c_clip_val\"}, \"width\": {\"field\": \"c_width_numeric_val\"}}}, \"type\": \"group\", \"scales\": [{\"name\": \"x\", \"type\": \"linear\", \"domain\": {\"data\": \"new_data\", \"fields\": [\"left\", \"right\"], \"sort\": true}, \"range\": [0, {\"signal\": \"width\"}], \"nice\": true, \"zero\": true}, {\"name\": \"y\", \"type\": \"linear\", \"domain\": {\"data\": \"new_data\", \"field\": \"count\"}, \"range\": [{\"signal\": \"height\"}, 0], \"nice\": true, \"zero\": true}], \"axes\": [{\"title\": \"Values\", \"scale\": \"x\", \"labelOverlap\": true, \"orient\": \"bottom\", \"tickCount\": {\"signal\": \"ceil(width/40)\"}, \"zindex\": 1}, {\"scale\": \"x\", \"domain\": false, \"grid\": true, \"labels\": false, \"maxExtent\": 0, \"minExtent\": 0, \"orient\": \"bottom\", \"tickCount\": {\"signal\": \"ceil(width/40)\"}, \"ticks\": false, \"zindex\": 0, \"gridScale\": \"y\"}, {\"title\": \"Count\", \"scale\": \"y\", \"labelOverlap\": true, \"orient\": \"left\", \"tickCount\": {\"signal\": \"ceil(height/40)\"}, \"zindex\": 1}, {\"scale\": \"y\", \"domain\": false, \"grid\": true, \"labels\": false, \"maxExtent\": 0, \"minExtent\": 0, \"orient\": \"left\", \"tickCount\": {\"signal\": \"ceil(height/40)\"}, \"ticks\": false, \"zindex\": 0, \"gridScale\": \"x\"}], \"style\": \"cell\", \"signals\": [{\"name\": \"width\", \"update\": \"250\"}, {\"name\": \"height\", \"update\": \"150\"}], \"marks\": [{\"name\": \"marks\", \"type\": \"rect\", \"style\": [\"rect\"], \"from\": {\"data\": \"new_data\"}, \"encode\": {\"hover\": {\"fill\": {\"value\": \"#7EC2F3\"}}, \"update\": {\"x\": {\"scale\": \"x\", \"field\": \"left\"}, \"x2\": {\"scale\": \"x\", \"field\": \"right\"}, \"y\": {\"scale\": \"y\", \"field\": \"count\"}, \"y2\": {\"scale\": \"y\", \"value\": 0}, \"fill\": {\"value\": \"#108EE9\"}}}}]}, {\"from\": {\"facet\": {\"name\": \"data_5\", \"data\": \"data_2\", \"field\": \"categorical\"}}, \"encode\": {\"enter\": {\"x\": {\"value\": 170}, \"width\": {\"value\": 250}, \"y\": {\"field\": \"graph_offset_categorical\"}, \"height\": {\"value\": 150}, \"fill\": {\"value\": \"#ffffff\"}, \"fillOpacity\": {\"value\": 0}, \"stroke\": {\"value\": \"#000000\"}, \"strokeWidth\": {\"value\": 0}}, \"update\": {\"x\": {\"signal\": \"datum[\\\"c_x_axis_back\\\"]+137\"}, \"clip\": {\"field\": \"c_clip_val_cat\"}, \"width\": {\"field\": \"c_width_numeric_val_cat\"}}}, \"type\": \"group\", \"style\": \"cell\", \"signals\": [{\"name\": \"unit\", \"value\": {}, \"on\": [{\"events\": \"mousemove\", \"update\": \"isTuple(group()) ? group() : unit\"}]}, {\"name\": \"pts\", \"update\": \"data(\\\"pts_store\\\").length &amp;&amp; {count: data(\\\"pts_store\\\")[0].values[0]}\"}, {\"name\": \"pts_tuple\", \"value\": {}, \"on\": [{\"events\": [{\"source\": \"scope\", \"type\": \"click\"}], \"update\": \"datum &amp;&amp; item().mark.marktype !== &apos;group&apos; ? {unit: \\\"\\\", encodings: [\\\"x\\\"], fields: [\\\"count\\\"], values: [datum[\\\"count\\\"]]} : null\", \"force\": true}]}, {\"name\": \"pts_modify\", \"on\": [{\"events\": {\"signal\": \"pts_tuple\"}, \"update\": \"modify(\\\"pts_store\\\", pts_tuple, true)\"}]}], \"marks\": [{\"name\": \"marks\", \"type\": \"rect\", \"style\": [\"bar\"], \"from\": {\"data\": \"data_5\"}, \"encode\": {\"hover\": {\"fill\": {\"value\": \"#7EC2F3\"}}, \"update\": {\"x\": {\"scale\": \"x\", \"field\": \"count\"}, \"x2\": {\"scale\": \"x\", \"value\": 0}, \"y\": {\"scale\": \"y\", \"field\": \"label\"}, \"height\": {\"scale\": \"y\", \"band\": true}, \"fill\": {\"value\": \"#108EE9\"}}}}], \"scales\": [{\"name\": \"x\", \"type\": \"linear\", \"domain\": {\"data\": \"data_5\", \"field\": \"count\"}, \"range\": [0, 250], \"nice\": true, \"zero\": true}, {\"name\": \"y\", \"type\": \"band\", \"domain\": {\"data\": \"data_5\", \"field\": \"label\", \"sort\": {\"op\": \"mean\", \"field\": \"label_idx\", \"order\": \"descending\"}}, \"range\": [150, 0], \"paddingInner\": 0.1, \"paddingOuter\": 0.05}], \"axes\": [{\"orient\": \"top\", \"scale\": \"x\", \"labelOverlap\": true, \"tickCount\": {\"signal\": \"ceil(width/40)\"}, \"title\": \"Count\", \"zindex\": 1}, {\"orient\": \"top\", \"scale\": \"x\", \"domain\": false, \"grid\": true, \"labels\": false, \"maxExtent\": 0, \"minExtent\": 0, \"tickCount\": {\"signal\": \"ceil(width/40)\"}, \"ticks\": false, \"zindex\": 0, \"gridScale\": \"y\"}, {\"scale\": \"y\", \"labelOverlap\": true, \"orient\": \"left\", \"title\": \"Label\", \"zindex\": 1}]}], \"type\": \"group\"}], \"type\": \"group\"}], \"config\": {\"axis\": {\"labelFont\": \"HelveticaNeue-Light, Arial\", \"labelFontSize\": 7, \"labelPadding\": 10, \"labelColor\": \"#595959\", \"titleFont\": \"HelveticaNeue-Light, Arial\", \"titleFontWeight\": \"normal\", \"titlePadding\": 9, \"titleFontSize\": 12, \"titleColor\": \"#595959\"}, \"axisY\": {\"minExtent\": 30}, \"style\": {\"rect\": {\"stroke\": \"rgba(200, 200, 200, 0.5)\"}, \"group-title\": {\"fontSize\": 20, \"font\": \"HelveticaNeue-Light, Arial\", \"fontWeight\": \"normal\", \"fill\": \"#595959\"}}}}";                                 var vega_json_parsed = JSON.parse(vega_json);                                 var toolTipOpts = {                                     showAllFields: true                                 };                                 if(vega_json_parsed["metadata"] != null){                                     if(vega_json_parsed["metadata"]["bubbleOpts"] != null){                                         toolTipOpts = vega_json_parsed["metadata"]["bubbleOpts"];                                     };                                 };                                 vegaEmbed("#vis", vega_json_parsed).then(function (result) {                                     vegaTooltip.vega(result.view, toolTipOpts);                                  });                             </script>                         </body>                     </html>' src="demo_iframe_srcdoc.htm">                         <p>Your browser does not support iframes.</p>                     </iframe>                 </body>             </html>



```python
dog_distances
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-automobile</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-bird</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-cat</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">dog-dog</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.95797614571203</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.75386473035126</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.419607706754384</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.47735903726335</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">46.00213318067788</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.3382958924861</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.83532688735542</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">32.84584956840554</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.946229069238804</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.615759085289056</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">36.97634108541546</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.03970731890584</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.68660600484793</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.08922699538214</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.575007291446106</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">33.90103276968193</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.22696649347584</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.27228869398105</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.77882479101661</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.484925090925636</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.58451176980721</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.146208923590486</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.11715782924591</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">34.94516534398124</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.10673529610854</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.523040105962316</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.60958309132649</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.095727834463545</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.32211409739762</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.19479183926956</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">39.90368673062214</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">37.76961310322034</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">41.82446549950164</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">40.156713166131446</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">38.067470016821176</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">35.10891446032838</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.497692940110376</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">45.55979626027668</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">42.72587329506032</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">43.242283258453455</td>
    </tr>
</table>
[1000 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>




```python
def is_dog_correct(row):
    if (row['dog-dog'] > row['dog-cat'] or
    row['dog-dog'] > row['dog-bird'] or
    row['dog-dog'] > row['dog-automobile']):
        return 0
    else:
        return 1
```


```python
is_dog_correct(dog_distances[0])
```




    1




```python
dog_distances[0]
```




    {'dog-automobile': 41.95797614571203,
     'dog-bird': 41.75386473035126,
     'dog-cat': 36.419607706754384,
     'dog-dog': 33.47735903726335}




```python
dog_distances.apply(is_dog_correct).sum()
```




    678




```python

```
