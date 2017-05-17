# gonn (gene ontology neural network)

Usage with the provided sample data:
python gonn.py -t training.fasta -c control.fasta

gonn is a basic TensorFlow neural network used to identify patterns in amino acid sequences indicative of a protein's molecular function. Using a small dataset it achieved an accuracy of 65%, indicating the potential for somewhat reliable identification with larger datasets.

Rather than creating a recurrent network which would be very data-intensive, frequencies of ordered pairs of amino acids are recorded. In theory this gives the neural network at least some capacity to react to sequence ordering, as the frequency of ordered pairs can be used to calculate the probability of their occurence in direct sequence.


Tested with data from the following uniprot searches:

http://www.uniprot.org/uniprot/?query=goa:(%22DNA%20binding%20[3677]%22)&fil=organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22+AND+reviewed%3Ayes&sort=score

http://www.uniprot.org/uniprot/?query=goa%3A%28NOT+%22DNA+binding+%5B3677%5D%22%29+AND+reviewed%3Ayes+AND+organism%3A%22Homo+sapiens+%28Human%29+%5B9606%5D%22&sort=score
