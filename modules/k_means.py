import tensorflow as tf 

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def k_means(vector_values, num_clusters, num_steps):
    print()
    print('Starting k-means')

    # Store vectors as 2D tensors
    vectors = tf.constant(vector_values)

    # The centroids, initialized with random vectors 
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), 
                                        [0, 0], [num_clusters, -1]))    # extract a slice of size [num_clusters, -1] starting at the location [0, 0]

    # extens the vectors to 3 dimensions for element-wise subtraction
    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    #
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
    assignments = tf.argmin(distances, 0)

    # gather all members of each cluster, average them to calculate new centroids for each cluster
    means = tf.concat(axis=0, 
                      values=[tf.reduce_mean(tf.gather(vectors, 
                                                       tf.reshape(tf.where(tf.equal(assignments, c)), 
                                                                  [1, -1])), 
                                             reduction_indices=[1]) for c in xrange(num_clusters)]
                      )

    # create an op which assigns the means to the centroids
    update_centroids = tf.assign(centroids, means)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        for step in xrange(num_steps):
            _, centroid_values, assignment_values = sess.run([update_centroids,
                                                              centroids,
                                                              assignments])
        print('Centroids:')
        print(centroid_values)

        data = {'x': [], 'y': [], 'cluster':[]}

        for i in xrange(len(assignment_values)):
            data['x'].append(vector_values[i][0])
            data['y'].append(vector_values[i][1])
            data['cluster'].append(assignment_values[i])

        df = pd.DataFrame(data)
    return df

def draw_data(data, emo_list):

    COLOR_LIST = ['c', 'm', 'b', 'y', 'r', 'g', 'k', ]

    # if FLAGS.emotions == 'basic3':
    #   N_EMO = 3
    #   emo_list = [('Neutral', 'y'), ('Angry', 'r'), ('Happy', 'c')]
    # elif FLAGS.emotions == 'basic4':
    #   N_EMO = 4
    #   emo_list = [('Neutral', 'y'), ('Angry', 'r'), ('Happy', 'c'), ('Scared', 'k')]
    # elif FLAGS.emotions == 'basic6':
    #   N_EMO = 6
    #   emo_list = [('Neutral', 'y'), ('Angry', 'r'), ('Happy', 'c'), ('Scared', 'k'), ('Sad', 'm'), ('Surprised', 'g')]
    # elif FLAGS.emotions == 'all':
    #   N_EMO = 9
    #   emo_list = [('Neutral', 'y'), ('Angry', 'r'), ('Happy', 'c'), ('Scared', 'k'), 
    #               ('Sad', 'm'), ('Excited', 'b'), ('Frustrated', ''), ('Surprised', 'g'), ('Disgusted', '')]
    # else:
    #   # Custom
    #   N_EMO = len(FLAGS.emotions.split())
    #   emo_list = list(zip(FLAGS.emotions.split(', '), COLOR_LIST[:N_EMO]))

    if FLAGS.emotions == 'basic3':
        N_EMO = 3
        emo_list = [('LA', 'y'), ('HN', 'r'), ('HP', 'c')]
    elif FLAGS.emotions == 'basic4':
        N_EMO = 4
        emo_list = [('LA', 'y'), ('HN', 'r'), ('HP', 'c')]
    elif FLAGS.emotions == 'basic6':
        N_EMO = 6
        emo_list = [('LA', 'y'), ('HN', 'r'), ('HP', 'c')]
    elif FLAGS.emotions == 'all':
        N_EMO = 9
        emo_list = [('LA', 'y'), ('HN', 'r'), ('HP', 'c')]
    else:
        # Custom
        N_EMO = len(FLAGS.emotions.split())
        emo_list = list(zip(FLAGS.emotions.split(', '), COLOR_LIST[:N_EMO]))

    fig = plt.figure()

    if FLAGS.dimensions == 'vad':
        ax = Axes3D(fig)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Activation')
        ax.set_zlabel('Dominance')
        
        for emo in emo_list:
            mask = data['emotion'] == emo[0]
            ax.scatter(xs=data[mask]['valence'],
                       ys=data[mask]['activation'], 
                       zs=data[mask]['dominance'], 
                       c=emo[1], zdir='z', 
                       label= emo[0] + ' (' + str(len(data[mask])) + ')')
    else:
        ax = fig.add_subplot(111)
        ax.set_xlabel('Valence')
        ax.set_ylabel('Activation')
        for emo in emo_list:
            mask = data['emotion'] == emo[0]
            ax.scatter(x=data[mask]['valence'],
                       y=data[mask]['activation'],  
                       c=emo[1], 
                       label= emo[0] + ' (' + str(len(data[mask])) + ')')
    ax.legend() 
    plt.show()