import logging
import inspect
import paho.mqtt.client as mqtt
import json

#  import the client1

logger = logging.getLogger(__name__)


class MqttClient:
    def __init__(self, device="I1"):
        # broker_address = "iot.eclipse.org"
        broker_address = "localhost"
        self.client = mqtt.Client(device)  # create new instance
        self.client.on_message = self.on_message
        self.client.connect(broker_address)  # connect to broker

    def publish(self, topic, message):
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            self.client.publish(topic, message)  # publish
        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    # async def subscribe(self, topic):
    def subscribe(self, topic):
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            self.client.loop_start()
            self.client.subscribe(topic)
            # time.sleep(1)
            # logger.info(message)
            # await asyncio.sleep(2)
            # logger.info("after getting", message)
            self.client.loop_stop()
        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")

    def on_message(self, client, user_data, message):
        try:
            _module_name = inspect.currentframe().f_code.co_name
            logger.info(f"Starting {_module_name} module")
            logger.info(str(message.payload.decode("utf-8")))
            param = str(message.payload.decode("utf-8"))
            param = json.loads(param)
            ConfigValuesObj.update_config(param)

        except Exception as error:
            logger.error(f"Error in {_module_name} module, {error}")
        finally:
            logger.info(f"Returning from {_module_name} module")