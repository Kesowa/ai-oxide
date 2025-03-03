const ampq = require("amqplib/callback_api");
const req = {
  file: "raster/1581c5f7-1041-4c89-83ec-eaf272bdc1f9.zip",
  metadata: {
    mission_id: "663f4c81d06158a3d86deb13",
    layer_id: "671f5452c3660e6442ccdbe5",
    user_id: "62663db5beb931c91c17fa58",
    tenant_id: "6103cf49fdcf5ebaca4a3893",
  },
};
const msg = JSON.stringify(req);

ampq.connect("amqp://localhost:5672", function(error, connection) {
  if (error) {
    throw error;
  }
  connection.createChannel(function(err, channel) {

    channel.assertQueue("file.decompress.req", {
      durable: true
    });

    channel.sendToQueue("file.decompress.req", Buffer.from(msg), {
      persistent: true,
      contentType: "application/json",
      type: "video.transcode",
    });
    console.log(" [X] Sent %s", msg);
  });

  connection.createChannel(function(err, channel) {
    channel.assertQueue("file.decompress.res", {
      durable: true
    });

    channel.consume("file.decompress.res", function(msg) {
      console.log("received: ", JSON.parse(msg.content));
      channel.ack(msg);
      setTimeout(function() {
        connection.close();
        process.exit(0);
      }, 500);
    });
  });
});
