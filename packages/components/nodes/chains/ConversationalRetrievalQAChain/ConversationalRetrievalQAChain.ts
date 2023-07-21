import { BaseLanguageModel } from 'langchain/base_language'
import { ICommonObject, IMessage, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses } from '../../../src/utils'
import { ConversationalRetrievalQAChain, ConversationalRetrievalQAChainInput, LLMChain, QAChainParams, loadQAChain } from 'langchain/chains'
import { AIMessage, ChainValues, HumanMessage } from 'langchain/schema'
import { BaseRetriever } from 'langchain/schema/retriever'
import { BaseChatMemory, BufferMemory, ChatMessageHistory } from 'langchain/memory'
import { PromptTemplate } from 'langchain/prompts'
import { ConsoleCallbackHandler, CustomChainHandler } from '../../../src/handler'
import {
    default_map_reduce_template,
    default_qa_template,
    qa_template,
    map_reduce_template,
    CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT,
    default_convo_chain_prompt,
    convo_chain_prompt
} from './prompts'
import { CallbackManagerForChainRun } from 'langchain/callbacks'

class ConversationalRetrievalQAChain_Chains implements INode {
    label: string
    name: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    description: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'Conversational Retrieval QA Chain'
        this.name = 'conversationalRetrievalQAChain'
        this.type = 'ConversationalRetrievalQAChain'
        this.icon = 'chain.svg'
        this.category = 'Chains'
        this.description = 'Document QA - built on RetrievalQAChain to provide a chat history component'
        this.baseClasses = [this.type, ...getBaseClasses(ConversationalRetrievalQAChain)]
        this.inputs = [
            {
                label: 'Language Model',
                name: 'model',
                type: 'BaseLanguageModel'
            },
            {
                label: 'Vector Store Retriever',
                name: 'vectorStoreRetriever',
                type: 'BaseRetriever'
            },
            {
                label: 'Long Term Memory',
                name: 'memory',
                type: 'DynamoDBChatMemory | RedisBackedChatMemory | ZepMemory',
                optional: true,
                description: 'Only accept long term memory. If none connected, a default BufferMemory will be used'
            },
            {
                label: 'Return Source Documents',
                name: 'returnSourceDocuments',
                type: 'boolean',
                optional: true
            },
            {
                label: 'System Message',
                name: 'systemMessagePrompt',
                type: 'string',
                rows: 4,
                additionalParams: true,
                optional: true,
                placeholder:
                    'I want you to act as a document that I am having a conversation with. Your name is "AI Assistant". You will provide me with answers from the given info. If the answer is not included, say exactly "Hmm, I am not sure." and stop after that. Refuse to answer any question not about the info. Never break character.'
            },
            {
                label: 'Chain Option',
                name: 'chainOption',
                type: 'options',
                options: [
                    {
                        label: 'MapReduceDocumentsChain',
                        name: 'map_reduce',
                        description:
                            'Suitable for QA tasks over larger documents and can run the preprocessing step in parallel, reducing the running time'
                    },
                    {
                        label: 'RefineDocumentsChain',
                        name: 'refine',
                        description: 'Suitable for QA tasks over a large number of documents.'
                    },
                    {
                        label: 'StuffDocumentsChain',
                        name: 'stuff',
                        description: 'Suitable for QA tasks over a small number of documents.'
                    }
                ],
                additionalParams: true,
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const model = nodeData.inputs?.model as BaseLanguageModel
        const vectorStoreRetriever = nodeData.inputs?.vectorStoreRetriever as BaseRetriever
        const systemMessagePrompt = nodeData.inputs?.systemMessagePrompt as string
        const returnSourceDocuments = nodeData.inputs?.returnSourceDocuments as boolean
        const chainOption = nodeData.inputs?.chainOption as string
        const memory = nodeData.inputs?.memory

        const obj: any = {
            verbose: process.env.DEBUG === 'true' ? true : false,
            questionGeneratorChainOptions: {
                template: CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT
            }
        }
        if (returnSourceDocuments) obj.returnSourceDocuments = returnSourceDocuments
        if (chainOption === 'map_reduce') {
            obj.qaChainOptions = {
                type: 'map_reduce',
                combinePrompt: PromptTemplate.fromTemplate(
                    systemMessagePrompt ? `${systemMessagePrompt}\n${map_reduce_template}` : default_map_reduce_template
                )
            }
        } else if (chainOption === 'refine') {
            // TODO: Add custom system message
        } else {
            obj.qaChainOptions = {
                type: 'stuff',
                prompt: PromptTemplate.fromTemplate(systemMessagePrompt ? `${systemMessagePrompt}\n${qa_template}` : default_qa_template)
            }
        }

        if (memory) {
            memory.inputKey = 'question'
            memory.outputKey = 'text'
            memory.memoryKey = 'chat_history'
            obj.memory = memory
        } else {
            obj.memory = new BufferMemory({
                memoryKey: 'chat_history',
                inputKey: 'question',
                outputKey: 'text',
                returnMessages: true
            })
        }

        obj.conversationalLLM = model
        obj.systemMessagePrompt = systemMessagePrompt

        const chain = ConvoChain.fromLLM(model, vectorStoreRetriever, obj)
        return chain
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | ICommonObject> {
        const chain = nodeData.instance as ConvoChain
        const returnSourceDocuments = nodeData.inputs?.returnSourceDocuments as boolean
        const memory = nodeData.inputs?.memory

        let model = nodeData.inputs?.model

        // Temporary fix: https://github.com/hwchase17/langchainjs/issues/754
        model.streaming = false
        chain.questionGeneratorChain.llm = model

        const obj = { question: input }

        // If external memory like Zep, Redis is being used, ignore below
        if (!memory && chain.memory && options && options.chatHistory) {
            const chatHistory = []
            const histories: IMessage[] = options.chatHistory
            const memory = chain.memory as BaseChatMemory

            for (const message of histories) {
                if (message.type === 'apiMessage') {
                    chatHistory.push(new AIMessage(message.message))
                } else if (message.type === 'userMessage') {
                    chatHistory.push(new HumanMessage(message.message))
                }
            }
            memory.chatHistory = new ChatMessageHistory(chatHistory)
            chain.memory = memory
        }

        const loggerHandler = new ConsoleCallbackHandler(options.logger)

        if (options.socketIO && options.socketIOClientId) {
            const handler = new CustomChainHandler(options.socketIO, options.socketIOClientId, undefined, returnSourceDocuments)
            const res = await chain.call(obj, [loggerHandler, handler])
            if (res.text && res.sourceDocuments) return res
            return res?.text
        } else {
            const res = await chain.call(obj, [loggerHandler])
            if (res.text && res.sourceDocuments) return res
            return res?.text
        }
    }
}

interface ConverInput {
    conversationalLLM: BaseLanguageModel
    systemMessagePrompt?: string
}

class ConvoChain extends ConversationalRetrievalQAChain {
    conversationalLLM?: BaseLanguageModel
    systemMessagePrompt?: string

    constructor(fields: ConversationalRetrievalQAChainInput & Partial<ConverInput>) {
        super(fields)
        this.conversationalLLM = fields.conversationalLLM
        this.systemMessagePrompt = fields.systemMessagePrompt
    }

    static fromLLM(
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        options: {
            outputKey?: string // not used
            returnSourceDocuments?: boolean
            qaChainOptions?: QAChainParams
            questionGeneratorChainOptions?: {
                llm?: BaseLanguageModel
                template?: string
            }
        } & Partial<ConverInput> &
            Omit<ConversationalRetrievalQAChainInput, 'retriever' | 'combineDocumentsChain' | 'questionGeneratorChain'> = {}
    ): ConversationalRetrievalQAChain {
        const {
            qaChainOptions = {
                type: 'stuff',
                prompt: undefined
            },
            questionGeneratorChainOptions,
            verbose,
            ...rest
        } = options

        const qaChain = loadQAChain(llm, qaChainOptions)

        const questionGeneratorChainPrompt = PromptTemplate.fromTemplate(
            questionGeneratorChainOptions?.template ?? CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT
        )
        const questionGeneratorChain = new LLMChain({
            prompt: questionGeneratorChainPrompt,
            llm: questionGeneratorChainOptions?.llm ?? llm,
            verbose
        })
        const instance = new this({
            retriever,
            combineDocumentsChain: qaChain,
            questionGeneratorChain,
            verbose,
            conversationalLLM: options.conversationalLLM,
            systemMessagePrompt: options.systemMessagePrompt,
            ...rest
        })
        return instance
    }

    // @ts-ignore
    async _call(values: ChainValues, runManager?: CallbackManagerForChainRun): Promise<ChainValues> {
        if (!(this.inputKey in values)) {
            throw new Error(`Question key ${this.inputKey} not found.`)
        }
        if (!(this.chatHistoryKey in values)) {
            throw new Error(`Chat history key ${this.chatHistoryKey} not found.`)
        }
        const question: string = values[this.inputKey]
        const chatHistory: string = ConversationalRetrievalQAChain.getChatHistoryString(values[this.chatHistoryKey])
        let newQuestion = question
        if (chatHistory.length > 0) {
            const result = await this.questionGeneratorChain.call(
                {
                    question,
                    chat_history: chatHistory
                },
                runManager?.getChild('question_generator')
            )
            const keys = Object.keys(result)
            if (keys.length === 1) {
                newQuestion = result[keys[0]]
            } else {
                throw new Error('Return from llm chain has multiple values, only single values supported.')
            }
        }
        const docs = await this.retriever.getRelevantDocuments(newQuestion, runManager?.getChild('retriever'))
        if (docs.length) {
            const inputs = {
                question: newQuestion,
                input_documents: docs,
                chat_history: chatHistory
            }
            const result = await this.combineDocumentsChain.call(inputs, runManager?.getChild('combine_documents'))

            console.log('combineDocumentsChain LMMMMMMMMMMMMMM =', result)

            if (this.returnSourceDocuments) {
                return {
                    ...result,
                    sourceDocuments: docs
                }
            }
            return result
        } else {
            const convoChain = new LLMChain({
                llm: this.conversationalLLM as BaseLanguageModel,
                prompt: PromptTemplate.fromTemplate(
                    this.systemMessagePrompt ? `${this.systemMessagePrompt}\n${convo_chain_prompt}` : default_convo_chain_prompt
                ),
                verbose: process.env.DEBUG === 'true' ? true : false
            })

            const inputs = {
                question,
                chat_history: chatHistory
            }

            const result = await convoChain.call(inputs, runManager?.getChild('combine_documents'))

            console.log('CONVERSATION LMMMMMMMMMMMMMM =', result)

            if (this.returnSourceDocuments) {
                return {
                    ...result,
                    sourceDocuments: docs
                }
            }
            return result
        }
    }
}

module.exports = { nodeClass: ConversationalRetrievalQAChain_Chains }
