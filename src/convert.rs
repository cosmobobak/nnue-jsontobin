
use serde::Deserialize;

const INPUT_SIZE: usize = 768;

// example of the json format:
// {
//     "perspective.weight": [ ... ],
//     "perspective.bias": [ ... ],
//     "factoriser.weight": [ ... ],
//     "factoriser.bias": [ ... ],
//     "out.weight": [ ... ],
//     "out.bias": [ ... ]
// }
// the "factoriser" fields are optional, and will be ignored if they are not present.
// we also accept aliases for the field names, e.g. "ft.weight" instead of "perspective.weight",
// and "fft.weight" instead of "factoriser.weight".

#[derive(Deserialize)]
struct FullNetworkWeights {
    #[serde(rename = "perspective.weight")]
    #[serde(alias = "ft.weight")]
    perspective_weight: Vec<Vec<f64>>,
    #[serde(rename = "perspective.bias")]
    #[serde(alias = "ft.bias")]
    perspective_bias: Vec<f64>,
    #[serde(rename = "factoriser.weight")]
    #[serde(alias = "fft.weight")]
    factoriser_weight: Option<Vec<Vec<f64>>>,
    #[serde(rename = "factoriser.bias")]
    #[serde(alias = "fft.bias")]
    factoriser_bias: Option<Vec<f64>>,
    #[serde(rename = "out.weight")]
    out_weight: Vec<Vec<f64>>,
    #[serde(rename = "out.bias")]
    out_bias: Vec<f64>,
}

pub struct QuantisedMergedNetwork {
    pub feature_weights: Vec<i16>,
    pub feature_bias: Vec<i16>,
    pub output_weights: Vec<i16>,
    pub output_bias: Vec<i16>,
}

fn extract_weights(weights: &[Vec<f64>], weight_array: &mut [i16], stride: usize, k: i32, flip: bool) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, output) in weights.iter().enumerate() {
        for (j, weight) in output.iter().enumerate() {
            let index = if flip { j * stride + i } else { i * stride + j };
            weight_array[index] = (weight * f64::from(k)) as i16;
        }
    }
}

fn extract_biases(biases: &[f64], bias_array: &mut [i16], k: i32) {
    #![allow(clippy::cast_possible_truncation)]
    for (i, bias) in biases.iter().enumerate() {
        bias_array[i] = (bias * f64::from(k)) as i16;
    }
}

pub fn from_json(
    json: &str,
    qa: i32,
    qb: i32,
) -> Result<QuantisedMergedNetwork, Box<dyn std::error::Error>> {
    let network_weights: FullNetworkWeights = serde_json::from_str(json)?;

    // check that the arrays are the right size
    let neurons = if let Some(factoriser_weight) = &network_weights.factoriser_weight {
        // if we have a factoriser, it gives us the correct number of neurons
        factoriser_weight.len()
    } else {
        // otherwise, there aren't any buckets, so we can safely use the perspective weight
        network_weights.perspective_weight.len()
    };
    println!("ft_size: {}", network_weights.perspective_weight.len());
    println!("neurons: {neurons}");
    let out_size = network_weights.out_weight[0].len();
    println!("out_size: {out_size}");
    if 2 * neurons != out_size {
        return Err(format!(
            "there are {neurons} neurons, but out.weight has {out_size} inputs (should be twice as many)"
        ).into());
    }

    println!("Hope you're using a {neurons}x2 net, because that's what this looks like to me!");

    let buckets = if let Some(factoriser) = &network_weights.factoriser_weight {
        // buckets is the number of times that a factoriser neuron can fit into a perspective neuron
        network_weights.perspective_weight[0].len() / factoriser[0].len()
    } else {
        // if there's no factoriser, there's essentially one bucket.
        1
    };

    if buckets == 1 {
        println!("This net doesn't have any buckets. (or it has one bucket, in which case wtf are you doing?)");
    } else {
        println!("There are {buckets} buckets in this net (we think).");
    }

    // allocate buffers for the weights and biases
    let mut feature_weights_buf = vec![0i16; neurons * INPUT_SIZE * buckets];
    let mut feature_bias_buf = vec![0i16; neurons * buckets];
    let mut factoriser_weights_buf = vec![0i16; neurons * INPUT_SIZE];
    let mut factoriser_bias_buf = vec![0i16; neurons];
    let mut output_weights_buf = vec![0i16; out_size];
    let mut output_bias_buf = vec![0i16; 1];

    // read the weights and biases into the buffers
    extract_weights(&network_weights.perspective_weight, &mut feature_weights_buf, neurons, qa, true);
    extract_biases(&network_weights.perspective_bias, &mut feature_bias_buf, qa);
    extract_weights(&network_weights.out_weight, &mut output_weights_buf, out_size, qb, false);
    extract_biases(&network_weights.out_bias, &mut output_bias_buf, qa * qb);

    if let Some(factoriser_weight) = &network_weights.factoriser_weight {
        // if we got a factoriser, read it into the buffer
        extract_weights(factoriser_weight, &mut factoriser_weights_buf, neurons, qa, true);
        // then add it to each bucket:
        let chunk_size = network_weights.perspective_weight.len() / buckets;
        for subnet in feature_weights_buf.chunks_mut(chunk_size) {
            for (src, tgt) in factoriser_weights_buf.iter().zip(subnet.iter_mut()) {
                *tgt += *src;
            }
        }
    }

    if let Some(factoriser_bias) = &network_weights.factoriser_bias {
        // if we got a factoriser, read it into the buffer
        extract_biases(factoriser_bias, &mut factoriser_bias_buf, qa);
        // then add it to each bucket:
        let chunk_size = network_weights.perspective_bias.len() / buckets;
        for subnet in feature_bias_buf.chunks_mut(chunk_size) {
            for (src, tgt) in factoriser_bias_buf.iter().zip(subnet.iter_mut()) {
                *tgt += *src;
            }
        }
    }

    // return the buffers
    Ok(QuantisedMergedNetwork {
        feature_weights: feature_weights_buf,
        feature_bias: feature_bias_buf,
        output_weights: output_weights_buf,
        output_bias: output_bias_buf,
    })
}

mod tests {
    #[test]
    fn test_from_json_0030() {
        use crate::convert::QuantisedMergedNetwork;
        let json = std::fs::read_to_string("validation/net0030/viri0030.json").unwrap();
        let QuantisedMergedNetwork { feature_weights, feature_bias, output_weights, output_bias } =
            crate::convert::from_json(&json, 255, 64).unwrap();
        assert_eq!(feature_weights.len(), 768 * 256);
        assert_eq!(feature_bias.len(), 256);
        assert_eq!(output_weights.len(), 512);
        assert_eq!(output_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0030/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0030/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0030/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0030/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0030/viri0030.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &feature_weights[..],
            &feature_bias[..],
            &output_weights[..],
            &output_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }

    #[test]
    fn test_from_json_0056() {
        use crate::convert::QuantisedMergedNetwork;
        let json = std::fs::read_to_string("validation/net0056/viri0056.json").unwrap();
        let QuantisedMergedNetwork { feature_weights: ft_weights, feature_bias: ft_bias, output_weights: out_weights, output_bias: out_bias } =
            crate::convert::from_json(&json, 255, 64).unwrap();
        assert_eq!(ft_weights.len(), 768 * 512);
        assert_eq!(ft_bias.len(), 512);
        assert_eq!(out_weights.len(), 1024);
        assert_eq!(out_bias.len(), 1);
        let validation_ft_weight = std::fs::read("validation/net0056/feature_weights.bin").unwrap();
        let validation_ft_bias = std::fs::read("validation/net0056/feature_bias.bin").unwrap();
        let validation_o_weight = std::fs::read("validation/net0056/output_weights.bin").unwrap();
        let validation_o_bias = std::fs::read("validation/net0056/output_bias.bin").unwrap();
        let validation_binary = std::fs::read("validation/net0056/viri0056.bin").unwrap();

        let validation_concat = [
            &validation_ft_weight[..],
            &validation_ft_bias[..],
            &validation_o_weight[..],
            &validation_o_bias[..],
        ]
        .concat();

        assert_eq!(
            validation_binary, validation_concat,
            "validation binary is not as expected"
        );

        let our_concat = [
            &ft_weights[..],
            &ft_bias[..],
            &out_weights[..],
            &out_bias[..],
        ]
        .concat();
        let our_concat = unsafe {
            let inner = our_concat.align_to::<u8>().1;
            inner
        };

        assert_eq!(
            validation_binary, our_concat,
            "our binary is not as expected"
        );
    }
}
