package main

import (
	"encoding/csv"
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"io"
	"log"
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
	})[0].Value().([]int32)

	var votes map[int32]int
	votes = make(map[int32]int)
	for i := 0; i < len(predictions); i++ {
		votes[predictions[i]]++
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

func Learn(model *tg.Model, observations [32]Observation) float32 {
	var sensorData [32][1][3]float32
	var labels [32]int32

	for i := 0; i < 32; i++ {
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

func NewObservation(record []string, mapping map[string]int32) Observation {
	var err error
	var id int
	if id, err = strconv.Atoi(record[0]); err != nil {
		log.Fatal(err)
	}

	activity := strings.ToLower(record[1])

	var sec int
	if sec, err = strconv.Atoi(record[2]); err != nil {
		log.Fatal(err)
	}

	var x, y, z float64

	if x, err = strconv.ParseFloat(record[3], 32); err != nil {
		log.Fatal(err)
	}

	if y, err = strconv.ParseFloat(record[4], 32); err != nil {
		log.Fatal(err)
	}

	if z, err = strconv.ParseFloat(strings.TrimRight(record[5], ";"), 32); err != nil {
		log.Fatal(err)
	}

	return Observation{
		ID:        id,
		Activity:  activity,
		Label:     mapping[activity],
		Timestamp: time.Unix(int64(sec), 0),
		X:         float32(x),
		Y:         float32(y),
		Z:         float32(z),
	}
}

func main() {
	var err error
	var filePtr *os.File

	if filePtr, err = os.Open("WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"); err != nil {
		log.Fatal(err)
	}

	defer func() {
		if err := filePtr.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	mapping := map[string]int32{
		"walking":    0,
		"jogging":    1,
		"upstairs":   2,
		"downstairs": 3,
		"sitting":    4,
		"standing":   5,
	}

	model := tg.LoadModel("at", []string{"serve"}, nil)

	reader := csv.NewReader(filePtr)

	var record []string
	var read int
	// 33,Jogging,49106062271000,5.012288,11.264028,0.95342433;
	var user int
	var label int32

	const batchSize int = 32

	var observations [32]Observation

	for {
		record, err = reader.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		o := NewObservation(record, mapping)
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

		observations[read-1] = o

		if read == batchSize {
			loss := Learn(model, observations)
			read = 0
			log.Printf("[Go] loss %f\n", loss)
		}

	}
}
