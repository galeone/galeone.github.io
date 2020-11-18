package main

import (
	"encoding/csv"
	//"encoding/gob"
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// predict invokes the predict method of the model and returns the majority class
// predicted for the input batch.
//
// model: the *tg.Model to use
// sensorData: a tf.Tensor with shape (32, 1, 3)
func predict(model *tg.Model, sensorData *tf.Tensor) int32 {
	predictions := model.Exec([]tf.Output{
		model.Op("StatefulPartitionedCall_2", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("predict_sensor_data", 0): sensorData,
	})[0].Value().([][]int32)

	var votes map[int32]int
	votes = make(map[int32]int)
	for i := 0; i < len(predictions); i++ {
		votes[predictions[i][0]]++
	}

	var maxVote int = -1
	var topLabel int32 = 0
	for label, vote := range votes {
		if vote > maxVote {
			maxVote = vote
			topLabel = label
		}
	}
	return topLabel
}

func observationsToTensor(observations [batchSize]Observation) *tf.Tensor {

	var sensorData [batchSize][1][3]float32
	size := len(observations)
	if size < batchSize {
		log.Fatalf("Observations size %d < batch size %d", size, batchSize)
	}

	for i := 0; i < size; i++ {
		sensorData[i][0] = [3]float32{observations[i].X, observations[i].Y, observations[i].Z}
	}

	var err error
	var sensorTensor *tf.Tensor

	if sensorTensor, err = tf.NewTensor(sensorData); err != nil {
		log.Fatal(err)
	}

	return sensorTensor
}

func Predict(model *tg.Model, observations [batchSize]Observation) int32 {
	sensorData := observationsToTensor(observations)
	return predict(model, sensorData)
}

/*signature_def['learn']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['labels'] tensor_info:
        dtype: DT_INT32
        shape: (-1)
        name: learn_labels:0
    inputs['sensor_data'] tensor_info:
        dtype: DT_FLOAT
        shape: (32, 1, 3)
        name: learn_sensor_data:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['loss'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: StatefulPartitionedCall_1:0
  Method name is: tensorflow/serving/predict

signature_def['predict']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['sensor_data'] tensor_info:
        dtype: DT_FLOAT
        shape: (32, 1, 3)
        name: predict_sensor_data:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['predictions'] tensor_info:
        dtype: DT_INT32
        shape: (32)
        name: StatefulPartitionedCall_2:0
  Method name is: tensorflow/serving/predict
*/

const batchSize int = 32

// learn runs an optimization step on the model, using a batch of values
// coming from the same observed activity.
// NOTE: all the observations must have the same label in the same batch.
//
// model: the *tg.Model to use
// sensorData: a tf.Tensor with shape (32, 1, 3)
// labels: a tf.Tensor with shape (32)
func learn(model *tg.Model, sensorData *tf.Tensor, labels *tf.Tensor) float32 {
	loss := model.Exec([]tf.Output{
		model.Op("StatefulPartitionedCall_1", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("learn_sensor_data", 0): sensorData,
		model.Op("learn_labels", 0):      labels,
	})[0]

	return loss.Value().(float32)
}

func Learn(model *tg.Model, observations [batchSize]Observation) float32 {
	var sensorData [batchSize][1][3]float32
	var labels [batchSize]int32

	for i := 0; i < batchSize; i++ {
		sensorData[i][0] = [3]float32{observations[i].X, observations[i].Y, observations[i].Z}
		labels[i] = observations[i].Label
	}

	var err error
	var sensorTensor *tf.Tensor
	var labelTensor *tf.Tensor

	if sensorTensor, err = tf.NewTensor(sensorData); err != nil {
		log.Fatal(err)
	}
	if labelTensor, err = tf.NewTensor(labels); err != nil {
		log.Fatal(err)
	}

	return learn(model, sensorTensor, labelTensor)

}

type Observation struct {
	// 3,Jogging,49106062271000,5.012288,11.264028,0.95342433
	ID        int
	Activity  string
	Label     int32
	Timestamp time.Time
	X         float32
	Y         float32
	Z         float32
}

func NewObservation(record []string, mapping map[string]int32) (*Observation, error) {
	var err error
	var id int
	if id, err = strconv.Atoi(record[0]); err != nil {
		return nil, err
	}

	activity := strings.ToLower(record[1])

	var sec int
	if sec, err = strconv.Atoi(record[2]); err != nil {
		return nil, err
	}

	var x, y, z float64

	if x, err = strconv.ParseFloat(record[3], 32); err != nil {
		return nil, err
	}

	if y, err = strconv.ParseFloat(record[4], 32); err != nil {
		return nil, err
	}

	if z, err = strconv.ParseFloat(strings.TrimRight(record[5], ";"), 32); err != nil {
		return nil, err
	}

	return &Observation{
		ID:        id,
		Activity:  activity,
		Label:     mapping[activity],
		Timestamp: time.Unix(int64(sec), 0),
		X:         float32(x),
		Y:         float32(y),
		Z:         float32(z),
	}, nil
}

func train(model *tg.Model, datasetPath string, mapping map[string]int32) {
	var err error
	var filePtr *os.File

	if filePtr, err = os.Open(datasetPath); err != nil {
		log.Fatal(err)
	}

	defer func() {
		if err := filePtr.Close(); err != nil {
			log.Fatalf("close: %s", err)
		}
	}()

	reader := csv.NewReader(filePtr)

	var record []string
	var read int
	// 33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
	var user int
	var label int32

	var observations [32]Observation

	for {
		record, err = reader.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		var o *Observation
		if o, err = NewObservation(record, mapping); err != nil {
			log.Printf("Skipping observation %v", record)
			continue
		}
		read++
		// Set the user for the current set of observations
		// The user and the label must be the same for all the training batch
		if read == 1 {
			user = o.ID
			label = o.Label
		}

		if user != o.ID || label != o.Label {
			if user != o.ID {
				log.Printf("Changing user %d -> %d\n", user, o.ID)
			}
			if label != o.Label {
				log.Printf("Changing label %d -> %d\n", label, o.Label)
			}
			read = 1
			user = o.ID
			label = o.Label
		}

		observations[read-1] = *o

		if read == batchSize {
			Learn(model, observations)
			read = 0
		}
	}
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

func main() {
	mapping := map[string]int32{
		"walking":    0,
		"jogging":    1,
		"upstairs":   2,
		"downstairs": 3,
		"sitting":    4,
		"standing":   5,
	}

	var model tg.Model = *tg.LoadModel("at", []string{"serve"}, nil)

	// First train:
	// NOTE: the dataset has been manually fixed since thanks to
	// the error handlig in the CSV reading, we found out a wrong line
	// Try it yourself and find it too :)
	train(&model, "WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", mapping)

	// TODO: find a way to serialize the model.
	// Currently tensorflow/go and also tfgo do not support model serialization easily.
	// However, it should be possibile to use encoding/gob + the TensorFlow C API
	// to correctly serialize the tf.SavedModel status and restore it.

	// If we find a way to save a SavedModel in the SavedModel serialization format
	// then we can just use SavedModels to save/restore models using the TensorFlow API.
	// I guess this is the best way.

	// Otherwise, we can create a different file format, that's like a modifiable SavedModel
	// that starts from a real SavedModel with tf.Variable inside.
	// We load it in memory and train it (as we did in the previous line), and
	// after that we serialize in a binary file (or whatever) the SavedModel status.
	// After that, when we want to use and restore it, we reload from this file
	// (that's not a SavedModel, thus we can't use the TensorFlow C API) and use it.

	// With a trained model, we are ready to receive other observations
	// and use them to train it during its lifetime.

	observations := make(chan Observation, batchSize)
	prediction := make(chan int32, 1)

	go func() {
		var obs [batchSize]Observation
		for i := 0; i < batchSize; i++ {
			obs[i] = <-observations
		}

		prediction <- Predict(&model, obs)
		// requeue all the observations in the same channel
		// in order to let other routines to use the same
		// observations (for training with the predicted/adjusted label)
		for i := 0; i < batchSize; i++ {
			observations <- obs[i]
		}

	}()

	done := make(chan bool, 1)
	go func() {
		// This is an example of a real case
		// where the sensor is gathering some data
		// and the user labels it (perhaps after our model suggestion that
		// can be used as a hint for the user to set the label :) ).

		// This function generates the data of the activity
		// and sends a batch of this data in the observation channel.

		// On the other side of the channel we have the model waiting
		// for this data, that sends back a suggested label and waits for a
		// user confirmation.

		// Generate sensor data

		var min, max float32 = 1, 10
		rand.Seed(time.Now().UnixNano())

		for i := 0; i < batchSize; i++ {
			x := min + rand.Float32()*(max-min)
			y := min + rand.Float32()*(max-min)
			z := min + rand.Float32()*(max-min)

			observations <- Observation{
				Timestamp: time.Now(),
				X:         x,
				Y:         y,
				Z:         z,
			}
		}

		predictedLabel := <-prediction
		log.Printf("predicted label: %d", predictedLabel)

		// In the real case, the predicted label is asked for
		// confirmation to the user and if the user accepts the suggestion
		// the model is trained with the current set of the observations
		// and the confirmed label (or the new label given by the user)

		var obs [batchSize]Observation
		for i := 0; i < batchSize; i++ {
			obs[i] = <-observations
			obs[i].Label = predictedLabel
		}

		Learn(&model, obs)
		// Here we terminarte the application, but in the real scenario
		// this is an endless loop with frequent checkpoints
		// and model versioning.
		done <- true
	}()

	<-done

}
