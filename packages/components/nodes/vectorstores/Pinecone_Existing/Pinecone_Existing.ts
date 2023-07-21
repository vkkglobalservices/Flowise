import { INode, INodeData, INodeOutputsValue, INodeParams } from '../../../src/Interface'
import { PineconeClient } from '@pinecone-database/pinecone'
import { PineconeLibArgs, PineconeStore } from 'langchain/vectorstores/pinecone'
import { Embeddings } from 'langchain/embeddings/base'
import { Document } from 'langchain/document'
import { getBaseClasses } from '../../../src/utils'

class Pinecone_Existing_VectorStores implements INode {
    label: string
    name: string
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    outputs: INodeOutputsValue[]

    constructor() {
        this.label = 'Pinecone Load Existing Index'
        this.name = 'pineconeExistingIndex'
        this.type = 'Pinecone'
        this.icon = 'pinecone.png'
        this.category = 'Vector Stores'
        this.description = 'Load existing index from Pinecone (i.e: Document has been upserted)'
        this.baseClasses = [this.type, 'VectorStoreRetriever', 'BaseRetriever']
        this.inputs = [
            {
                label: 'Embeddings',
                name: 'embeddings',
                type: 'Embeddings'
            },
            {
                label: 'Pinecone Api Key',
                name: 'pineconeApiKey',
                type: 'password'
            },
            {
                label: 'Pinecone Environment',
                name: 'pineconeEnv',
                type: 'string'
            },
            {
                label: 'Pinecone Index',
                name: 'pineconeIndex',
                type: 'string'
            },
            {
                label: 'Pinecone Namespace',
                name: 'pineconeNamespace',
                type: 'string',
                placeholder: 'my-first-namespace',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Pinecone Metadata Filter',
                name: 'pineconeMetadataFilter',
                type: 'json',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Top K',
                name: 'topK',
                description: 'Number of top results to fetch. Default to 4',
                placeholder: '4',
                type: 'number',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Min Score',
                name: 'minScore',
                description: 'The minimum score (1-100) of results that will be used to build the reply',
                placeholder: '75',
                type: 'number',
                additionalParams: true,
                optional: true
            }
        ]
        this.outputs = [
            {
                label: 'Pinecone Retriever',
                name: 'retriever',
                baseClasses: this.baseClasses
            },
            {
                label: 'Pinecone Vector Store',
                name: 'vectorStore',
                baseClasses: [this.type, ...getBaseClasses(PineconeStore)]
            }
        ]
    }

    async init(nodeData: INodeData): Promise<any> {
        const pineconeApiKey = nodeData.inputs?.pineconeApiKey as string
        const pineconeEnv = nodeData.inputs?.pineconeEnv as string
        const index = nodeData.inputs?.pineconeIndex as string
        const pineconeNamespace = nodeData.inputs?.pineconeNamespace as string
        const minScore = nodeData.inputs?.minScore as string
        const pineconeMetadataFilter = nodeData.inputs?.pineconeMetadataFilter
        const embeddings = nodeData.inputs?.embeddings as Embeddings
        const output = nodeData.outputs?.output as string
        const topK = nodeData.inputs?.topK as string
        const k = topK ? parseInt(topK, 10) : 4

        const client = new PineconeClient()
        await client.init({
            apiKey: pineconeApiKey,
            environment: pineconeEnv
        })

        const pineconeIndex = client.Index(index)

        const obj: PineconeLibArgs & Partial<PineconeScore> = {
            pineconeIndex
        }

        if (pineconeNamespace) obj.namespace = pineconeNamespace
        if (pineconeMetadataFilter) {
            const metadatafilter = typeof pineconeMetadataFilter === 'object' ? pineconeMetadataFilter : JSON.parse(pineconeMetadataFilter)
            obj.filter = metadatafilter
        }
        if (minScore) {
            const minimumScore = parseInt(minScore, 10)
            obj.score = minimumScore / 100
        }

        const vectorStore = await PineconeExisting.fromExistingIndex(embeddings, obj)

        if (output === 'retriever') {
            const retriever = vectorStore.asRetriever(k)
            return retriever
        } else if (output === 'vectorStore') {
            ;(vectorStore as any).k = k
            return vectorStore
        }
        return vectorStore
    }
}

type PineconeMetadata = Record<string, any>

interface PineconeScore {
    score: number
}

class PineconeExisting extends PineconeStore {
    scoreThreshold?: number

    constructor(embeddings: Embeddings, args: PineconeLibArgs & Partial<PineconeScore>) {
        super(embeddings, args)
        this.scoreThreshold = args.score
    }

    static async fromExistingIndex(embeddings: Embeddings, dbConfig: PineconeLibArgs & Partial<PineconeScore>): Promise<PineconeStore> {
        const instance = new this(embeddings, dbConfig)
        return instance
    }

    // @ts-ignore
    async similaritySearchVectorWithScore(query: number[], k: number, filter?: PineconeMetadata): Promise<[Document, number][]> {
        if (filter && this.filter) {
            throw new Error('cannot provide both `filter` and `this.filter`')
        }
        const _filter = filter ?? this.filter
        const results = await this.pineconeIndex.query({
            queryRequest: {
                includeMetadata: true,
                namespace: this.namespace,
                topK: k,
                vector: query,
                filter: _filter
            }
        })

        const result: [Document, number][] = []

        if (results.matches) {
            for (const res of results.matches) {
                const { [this.textKey]: pageContent, ...metadata } = (res.metadata ?? {}) as PineconeMetadata
                if (res.score) {
                    console.log('SEARCHINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG = ', res.score)
                    if (this.scoreThreshold && res.score >= this.scoreThreshold) {
                        result.push([new Document({ metadata, pageContent }), res.score])
                    } else if (this.scoreThreshold && res.score < this.scoreThreshold) {
                        continue
                    } else {
                        result.push([new Document({ metadata, pageContent }), res.score])
                    }
                }
            }
        }

        return result
    }
}

module.exports = { nodeClass: Pinecone_Existing_VectorStores }
